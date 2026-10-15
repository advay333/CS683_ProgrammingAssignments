B=./bin/hashed_perceptron-no-no-no-no-no-no-no-lru-lru-lru-dyn_rrip-lru-lru-lru-lru-1core-t3nopf
for t in 1 2 3; do
  $B -warmup_instructions 25000000 -simulation_instructions 25000000 \
     -traces ./Trace_Files/Trace$t.gz > results_pf$t/t3nopf.txt &
done
wait