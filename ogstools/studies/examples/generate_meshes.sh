max=7
for (( i=0; i <= $max; ++i ))
do
    n=$((2**$i))
    generateStructuredMesh -e quad -o square_$i.vtu --lx 1 --ly 1 --lz 0 --nx $n --ny $n --nz 1
done
