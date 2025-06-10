echo "-------------------------------" > result.log
echo "-------------------------------" >> result.log
for n in {2..14}; do
    echo "Running with $n MPI rank(s)..."
    echo "Running with $n MPI rank(s)..." >> result.log
    mpirun -n "$n" special -mpi -python -s /Users/juanjose.garcia/dev/neurodamus/neurodamus/data/init.py --configFile=simulation_config.json --lb-mode=Memory > log

    grep "Memusage (RSS)" log | tail -n 1 >> result.log
    grep "Cell creation" log | tail -n 1 | awk -F'|' '{ gsub(/^[ \t]+/, "", $3); print "Cell creation: " $3 }' >> result.log
    grep "finished Run" log | tail -n 1 | awk -F'|' '{ gsub(/^[ \t]+/, "", $3); print "Run time: " $3 }' >> result.log
    echo "-------------------------------" >> result.log
    echo "-------------------------------" >> result.log
done

cat result.log