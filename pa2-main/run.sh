B=./bin/hashed_perceptron-no-spp-no-no-no-no-no-lru-lru-lru-lru-lru-lru-lru-lru-1core
for v in spp2 spp3; do for t in 2 3; do
  $B-$v -warmup_instructions 25000000 -simulation_instructions 25000000 \
        -traces ./Trace_Files/Trace$t.gz > results_pf$t/$v.txt
done; done