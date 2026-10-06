-- AgentCard experiment 1: synthetic sustained storage at the streamer seam.
-- No storage-device timing claim. See docs/agentcard/storage-experiment-001.md.
-- Original upstream RTL remains unmodified. All counts use one 10 ns clock.
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use std.env.all;

entity tb_agentcard_storage is
  generic (
    BANKS          : positive := 27;
    SERVICE_CYCLES : positive := 1;
    LATENCY        : natural := 8;
    SKEW           : natural := 0;
    PAGE_BEATS     : positive := 128;
    PAGE_DELAY     : natural := 0;
    FIFO_DEPTH     : positive := 64;
    BURST_BEATS    : positive := 16;
    OUTSTANDING    : positive := 16;
    CONSUMER_PERIOD: positive := 1;
    PAUSE_EVERY    : natural := 0;
    PAUSE_CYCLES   : natural := 0;
    WARM_GROUPS    : positive := 512;
    MEASURE_GROUPS : positive := 2048;
    DRAIN_GROUPS   : positive := 393;
    -- Negative control, used only by the runner's expected-failure test.
    MUTATION       : natural := 0;
    CHECK_DATA     : boolean := true
  );
end entity;

architecture sim of tb_agentcard_storage is
  constant NW : positive := 24;
  constant NPS : positive := 3;
  constant NP : positive := NW+NPS;
  constant DW : positive := 256;
  constant BB : positive := DW/8;
  constant TOTAL : positive := WARM_GROUPS+MEASURE_GROUPS+DRAIN_GROUPS;
  constant REGION : positive := 1048576;
  type ints is array(natural range <>) of integer;
  type queue is array(0 to NP-1, 0 to OUTSTANDING-1) of integer;
  signal clk : std_logic := '0';
  signal rst : std_logic := '1';
  signal start : std_logic := '0';
  signal cycle : natural := 0;
  signal arvalid, arready, rvalid, rready, rlast : std_logic_vector(NP-1 downto 0) := (others=>'0');
  signal araddr : std_logic_vector(NP*32-1 downto 0);
  signal arlen : std_logic_vector(NP*8-1 downto 0);
  signal arsize : std_logic_vector(NP*3-1 downto 0);
  signal arburst : std_logic_vector(NP*2-1 downto 0);
  signal rdata : std_logic_vector(NP*DW-1 downto 0) := (others=>'0');
  signal wb : std_logic_vector(NW*32-1 downto 0);
  signal sb : std_logic_vector(NPS*32-1 downto 0);
  signal wv, sv, wr, sr, demand : std_logic := '0';
  signal wd : std_logic_vector(48*32*4-1 downto 0);
  signal sd : std_logic_vector(48*16-1 downto 0);

  -- Source values vary by sequence, row and within-row coordinate.
  function nib(n, row, j : natural) return natural is
  begin
    return (n*7 + n/11 + row*3 + row/3 + j*5 + j/7 +
      (row/16)*(j+1) + (n/176)*(j+row+1) +
      (n/2816)*(j+1)*(row+1)) mod 16;
  end;
  function scale(n, row : natural) return natural is
  begin return (n*257 + row*1009 + n/17 + 123) mod 65536; end;

  -- Memory is encoded by global BIT offset within a 256-bit lane slice.
  function memory_word(p, n : natural) return std_logic_vector is
    variable v : std_logic_vector(DW-1 downto 0);
    variable x : unsigned(15 downto 0);
    variable bitpos, row, j, source_n, source_p : natural;
  begin
    source_n:=n; source_p:=p;
    if MUTATION=4 and p<NW and n=WARM_GROUPS+37 then source_p:=(p+12) mod NW; end if;
    if MUTATION=5 and p<NW and n=WARM_GROUPS+37 then source_n:=n-176; end if;
    if MUTATION=2 and n=WARM_GROUPS+37 then source_n:=n-BURST_BEATS; end if;
    for bitidx in 0 to DW-1 loop
      if p < NW then
        bitpos := source_p*DW+bitidx; row := bitpos/128; j := (bitpos mod 128)/4;
        x := to_unsigned(nib(source_n,row,j),16); v(bitidx) := x(bitpos mod 4);
      else
        bitpos := (p-NW)*DW+bitidx; row := bitpos/16;
        x := to_unsigned(scale(source_n,row),16); v(bitidx) := x(bitpos mod 16);
      end if;
    end loop;
    if MUTATION=1 and p=7 and n=WARM_GROUPS+37 then v(19) := not v(19); end if;
    if MUTATION=3 and p=NW+1 and n=WARM_GROUPS+37 then v(91):=not v(91); end if;
    return v;
  end;
begin
  assert BANKS<=NP and NP mod BANKS=0 report "banks must divide 27" severity failure;
  assert BURST_BEATS<=128 and 128 mod BURST_BEATS=0 report "burst must divide a 4 KiB page" severity failure;
  assert PAGE_BEATS mod BURST_BEATS=0 report "page must contain complete bursts" severity failure;
  assert FIFO_DEPTH>=BURST_BEATS+4 report "FIFO must admit a burst plus pipeline margin" severity failure;
  assert TOTAL*BB<REGION report "synthetic regions overlap" severity failure;
  assert WARM_GROUPS>FIFO_DEPTH+4 and MEASURE_GROUPS>4*(FIFO_DEPTH+4)
    report "window too short to exhaust initial inventory" severity failure;
  assert DRAIN_GROUPS>FIFO_DEPTH+BURST_BEATS+4 report "tail too short to exclude final prefetch drain" severity failure;
  assert PAUSE_EVERY=0 or PAUSE_CYCLES<PAUSE_EVERY report "consumer never resumes" severity failure;
  clk <= not clk after 5 ns;
  process begin
    wait for 40 ns; wait until falling_edge(clk); rst<='0'; start<='1';
    wait until falling_edge(clk); start<='0'; wait;
  end process;
  process(clk) begin if rising_edge(clk) then if rst='0' then cycle<=cycle+1; end if; end if; end process;
  gen_w: for p in 0 to NW-1 generate wb((p+1)*32-1 downto p*32)<=std_logic_vector(to_unsigned(p*REGION,32)); end generate;
  gen_s: for p in 0 to NPS-1 generate sb((p+1)*32-1 downto p*32)<=std_logic_vector(to_unsigned((p+NW)*REGION,32)); end generate;
  demand <= '1' when rst='0' and cycle mod CONSUMER_PERIOD=0 and
    (PAUSE_EVERY=0 or (cycle mod maximum(1,PAUSE_EVERY))>=PAUSE_CYCLES) else '0';
  wr <= demand and sv; sr <= demand and wv;
  dut: entity work.weight_streamer
    generic map(NPORTS_W=>NW,NPORTS_S=>NPS,AXI_DW=>DW,ROWS_IF=>48,BLK=>32,
      DEPTH=>FIFO_DEPTH,MAXB=>BURST_BEATS,MAXOUT=>OUTSTANDING,FAST_POP=>true,DUAL_CLK=>false)
    port map(clk=>clk,rst=>rst,start=>start,w_base=>wb,w_beats=>TOTAL,s_base=>sb,s_beats=>TOTAL,
      m_arvalid=>arvalid,m_arready=>arready,m_araddr=>araddr,m_arlen=>arlen,
      m_arsize=>arsize,m_arburst=>arburst,m_rvalid=>rvalid,m_rready=>rready,
      m_rdata=>rdata,m_rlast=>rlast,w_valid=>wv,w_data=>wd,w_ready=>wr,
      s_valid=>sv,s_data=>sd,s_ready=>sr);

  -- One bounded request FIFO per AXI lane. Each bank multiplexes its assigned
  -- lanes round-robin and holds a selected response until it is accepted.
  -- Requests can overlap latency; page delay is per page-start REQUEST, not
  -- a NAND controller, cache, serialized die busy time, or NVMe model.
  process(clk)
    variable qseq, qlen, qdue : queue := (others=>(others=>0));
    variable head, tail, count, requested, received : ints(0 to NP-1) := (others=>0);
    variable owner : ints(0 to BANKS-1) := (others=>-1);
    variable last_accept : ints(0 to BANKS-1) := (others=>-2000000);
    variable last_read_lane : ints(0 to NP-1) := (others=>-1);
    variable start_lane, input_lane : ints(0 to NP-1) := (others=>0);
    variable rr, available : ints(0 to BANKS-1) := (others=>0);
    variable rv, rl : std_logic_vector(NP-1 downto 0) := (others=>'0');
    variable rd : std_logic_vector(NP*DW-1 downto 0) := (others=>'0');
    variable p, addr, n, len, due, pagewait : integer;
    variable consumed, c, measurement_cycles, starved, ready_cycles, input_beats : integer := 0;
    variable start_inventory, end_inventory, inv, suminv, peak_inv, peak_lane, outstanding_peak : integer := 0;
    variable min_inv : integer := integer'high;
    variable accepted, qtotal, response_stalls, ar_stalls : integer := 0;
    variable first_measure, last_measure, first_group, last_read, warm_end : integer := -1;
    variable active : boolean := false;
  begin
    if rising_edge(clk) and rst='0' then
      c := cycle;
      -- Start on the edge AFTER exactly WARM_GROUPS accepted groups.
      -- Inventory is sampled BEFORE that edge's input/output handshakes.
      if consumed=WARM_GROUPS and not active then
        active := true; first_measure := c; start_inventory:=0;
        for lane in 0 to NP-1 loop
          start_lane(lane):=received(lane)-consumed;
          start_inventory:=start_inventory+start_lane(lane);
        end loop;
      end if;
      accepted:=0;
      for b in 0 to BANKS-1 loop
        p:=owner(b);
        if p>=0 then
          if rvalid(p)='1' and rready(p)='1' then
            assert c-last_accept(b)>=SERVICE_CYCLES report "bank bandwidth violation" severity failure;
            last_accept(b):=c;
            if active then input_lane(p):=input_lane(p)+1; end if;
            assert qseq(p,head(p))=received(p) report "response sequence mismatch" severity failure;
            received(p):=received(p)+1; accepted:=accepted+1; last_read:=c; last_read_lane(p):=c;
            qseq(p,head(p)):=qseq(p,head(p))+1;
            qlen(p,head(p)):=qlen(p,head(p))-1;
            if qlen(p,head(p))=0 then head(p):=(head(p)+1) mod OUTSTANDING; count(p):=count(p)-1; end if;
            rv(p):='0'; rl(p):='0'; owner(b):=-1;
            rr(b):=(p+BANKS) mod NP;
            available(b):=c+SERVICE_CYCLES;
          elsif active then response_stalls:=response_stalls+1;
          end if;
        end if;
      end loop;
      for lane in 0 to NP-1 loop
        if arvalid(lane)='1' and arready(lane)='1' then
          addr:=to_integer(unsigned(araddr((lane+1)*32-1 downto lane*32)));
          len:=to_integer(unsigned(arlen((lane+1)*8-1 downto lane*8)))+1;
          assert addr=lane*REGION+requested(lane)*BB report "AR address discontinuity" severity failure;
          assert (addr mod 4096)+len*BB<=4096 report "AR crosses 4 KiB boundary" severity failure;
          assert arsize((lane+1)*3-1 downto lane*3)="101" and arburst((lane+1)*2-1 downto lane*2)="01"
            report "wrong AXI beat geometry" severity failure;
          assert len<=BURST_BEATS and requested(lane)+len<=TOTAL and count(lane)<OUTSTANDING report "request overflow" severity failure;
          pagewait:=0; if requested(lane) mod PAGE_BEATS=0 then pagewait:=PAGE_DELAY; end if;
          due:=c+1+LATENCY+(lane mod 5)*SKEW+pagewait;
          qseq(lane,tail(lane)):=requested(lane); qlen(lane,tail(lane)):=len; qdue(lane,tail(lane)):=due;
          tail(lane):=(tail(lane)+1) mod OUTSTANDING; count(lane):=count(lane)+1;
          requested(lane):=requested(lane)+len;
        elsif active and arvalid(lane)='1' then ar_stalls:=ar_stalls+1;
        end if;
        if count(lane)<OUTSTANDING then arready(lane)<='1'; else arready(lane)<='0'; end if;
      end loop;
      -- Stage a response for the NEXT rising edge, with no extra idle cycle
      -- at SERVICE_CYCLES=1. Backpressure occupies the selected bank.
      for b in 0 to BANKS-1 loop
        if c=0 then rr(b):=b; end if;
        if owner(b)=-1 and c+1>=available(b) then
          for offset in 0 to NP/BANKS-1 loop
            p:=(rr(b)+offset*BANKS) mod NP;
            if p mod BANKS=b and count(p)>0 and qdue(p,head(p))<=c+1 then
              owner(b):=p; rv(p):='1'; rd((p+1)*DW-1 downto p*DW):=memory_word(p,qseq(p,head(p)));
              if qlen(p,head(p))=1 then rl(p):='1'; else rl(p):='0'; end if;
              exit;
            end if;
          end loop;
        end if;
      end loop;
      rvalid<=rv; rlast<=rl; rdata<=rd;
      if active then
        measurement_cycles:=measurement_cycles+1; input_beats:=input_beats+accepted;
        if demand='1' then
          ready_cycles:=ready_cycles+1;
          if wv/='1' or sv/='1' then starved:=starved+1; end if;
        end if;
      end if;
      assert (wv and wr)=(sv and sr) report "consumer pair split" severity failure;
      if wv='1' and wr='1' then
        assert consumed<TOTAL report "extra output" severity failure;
        -- Oracle walks ROWS, then elements, rather than source lane slices.
        if CHECK_DATA then
          for row in 0 to 47 loop
            for j in 0 to 31 loop
              assert unsigned(wd(row*128+j*4+3 downto row*128+j*4))=nib(consumed,row,j)
                report "weight mismatch group="&integer'image(consumed)&" row="&integer'image(row) severity failure;
            end loop;
            assert unsigned(sd(row*16+15 downto row*16))=scale(consumed,row)
              report "scale mismatch group="&integer'image(consumed)&" row="&integer'image(row) severity failure;
          end loop;
        end if;
        if first_group=-1 then first_group:=c; end if;
        consumed:=consumed+1;
        if consumed=WARM_GROUPS then warm_end:=c; end if;
      end if;
      inv:=0; qtotal:=0;
      for lane in 0 to NP-1 loop
        n:=received(lane)-consumed;
        assert n>=0 and n<=FIFO_DEPTH+4 report "accepted-minus-consumed bound" severity failure;
        assert requested(lane)>=received(lane) and requested(lane)-received(lane)<=FIFO_DEPTH+BURST_BEATS
          report "inflight bound" severity failure;
        inv:=inv+n; qtotal:=qtotal+requested(lane)-received(lane);
        if active then peak_lane:=maximum(peak_lane,n); end if;
      end loop;
      if active then
        suminv:=suminv+inv; min_inv:=minimum(min_inv,inv); peak_inv:=maximum(peak_inv,inv);
        outstanding_peak:=maximum(outstanding_peak,qtotal);
      end if;
      if active and consumed=WARM_GROUPS+MEASURE_GROUPS then
        last_measure:=c; end_inventory:=inv;
        for lane in 0 to NP-1 loop
          assert start_lane(lane)+input_lane(lane)=MEASURE_GROUPS+received(lane)-consumed
            report "per-lane window conservation" severity failure;
        end loop;
        assert start_inventory+input_beats=MEASURE_GROUPS*NP+end_inventory report "window conservation" severity failure;
        assert ready_cycles=MEASURE_GROUPS+starved report "demand accounting" severity failure;
        assert measurement_cycles=last_measure-first_measure+1 report "cycle accounting" severity failure;
        assert input_beats>MEASURE_GROUPS*NP-start_inventory-NP report "measurement not live" severity failure;
        report "MEASURE cycles="&integer'image(measurement_cycles)&" groups="&integer'image(MEASURE_GROUPS)&
          " input_beats="&integer'image(input_beats)&" demand_cycles="&integer'image(ready_cycles)&
          " starved_cycles="&integer'image(starved)&" start_inventory="&integer'image(start_inventory)&
          " end_inventory="&integer'image(end_inventory)&" min_inventory="&integer'image(min_inv)&
          " peak_inventory="&integer'image(peak_inv)&" sum_inventory="&integer'image(suminv)&
          " peak_lane="&integer'image(peak_lane)&" peak_inflight="&integer'image(outstanding_peak)&
          " response_stalls="&integer'image(response_stalls)&" ar_stalls="&integer'image(ar_stalls)&
          " first_cycle="&integer'image(first_measure)&" last_cycle="&integer'image(last_measure);
        active:=false;
      end if;
      if consumed=TOTAL then
        for lane in 0 to NP-1 loop
          assert last_read_lane(lane)>last_measure report "lane finished before measured window ended" severity failure;
          assert received(lane)=TOTAL and requested(lane)=TOTAL and count(lane)=0 report "final conservation" severity failure;
        end loop;
        assert inv=0 and last_measure>=0 and last_read>=first_measure report "drain/window incomplete" severity failure;
        report "PASS agentcard_storage total_groups="&integer'image(TOTAL)&" total_beats="&integer'image(TOTAL*NP)&
          " first_group_cycle="&integer'image(first_group)&" warm_end_cycle="&integer'image(warm_end)&
          " final_cycle="&integer'image(c)&" last_read_cycle="&integer'image(last_read);
        finish;
      end if;
      assert c<2000000 report "watchdog: did not complete" severity failure;
    end if;
  end process;
end architecture;
