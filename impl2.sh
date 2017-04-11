
rm -rf results_impl2
mkdir results_impl2

files=('amazon0312' 'msdoor' 'roadNet-CA' 'soc-LiveJournal1' 'USA-road-d.CAL' 'web-Google')	
for file_name in "${files[@]}"
do
./sssp --input test_graphs/"${file_name}".txt --bsize 256 --bcount 8 --output output.txt --method tpe --usemem no --sync incore --sort src >  results_impl2/"${file_name}"_256_8_incore_src.stat
./sssp --input test_graphs/"${file_name}".txt --bsize 384 --bcount 5 --output output.txt --method tpe --usemem no --sync incore --sort src >  results_impl2/"${file_name}"_384_5_incore_src.stat
./sssp --input test_graphs/"${file_name}".txt --bsize 512 --bcount 4 --output output.txt --method tpe --usemem no --sync incore --sort src > results_impl2/"${file_name}"_512_4_incore_src.stat
./sssp --input test_graphs/"${file_name}".txt --bsize 768 --bcount 2 --output output.txt --method tpe --usemem no --sync incore --sort src > results_impl2/"${file_name}"_768_2_incore_src.stat
./sssp --input test_graphs/"${file_name}".txt --bsize 1024 --bcount 2 --output output.txt --method tpe --usemem no --sync incore --sort src > results_impl2/"${file_name}"_1024_2_incore_src.stat


./sssp --input test_graphs/"${file_name}".txt --bsize 256 --bcount 8 --output output.txt --method tpe --usemem no --sync incore --sort dest >  results_impl2/"${file_name}"_256_8_incore_dest.stat
./sssp --input test_graphs/"${file_name}".txt --bsize 384 --bcount 5 --output output.txt --method tpe --usemem no --sync incore --sort dest >  results_impl2/"${file_name}"_384_5_incore_dest.stat
./sssp --input test_graphs/"${file_name}".txt --bsize 512 --bcount 4 --output output.txt --method tpe --usemem no --sync incore --sort dest > results_impl2/"${file_name}"_512_4_incore_dest.stat
./sssp --input test_graphs/"${file_name}".txt --bsize 768 --bcount 2 --output output.txt --method tpe --usemem no --sync incore --sort dest > results_impl2/"${file_name}"_768_2_incore_dest.stat
./sssp --input test_graphs/"${file_name}".txt --bsize 1024 --bcount 2 --output output.txt --method tpe --usemem no --sync incore --sort dest > results_impl2/"${file_name}"_1024_2_incore_dest.stat


./sssp --input test_graphs/"${file_name}".txt --bsize 256 --bcount 8 --output output.txt --method tpe --usemem no --sync outcore --sort src >  results_impl2/"${file_name}"_256_8_outcore_src.stat
./sssp --input test_graphs/"${file_name}".txt --bsize 384 --bcount 5 --output output.txt --method tpe --usemem no --sync outcore --sort src >  results_impl2/"${file_name}"_384_5_outcore_src.stat
./sssp --input test_graphs/"${file_name}".txt --bsize 512 --bcount 4 --output output.txt --method tpe --usemem no --sync outcore --sort src > results_impl2/"${file_name}"_512_4_outcore_src.stat
./sssp --input test_graphs/"${file_name}".txt --bsize 768 --bcount 2 --output output.txt --method tpe --usemem no --sync outcore --sort src > results_impl2/"${file_name}"_768_2_outcore_src.stat
./sssp --input test_graphs/"${file_name}".txt --bsize 1024 --bcount 2 --output output.txt --method tpe --usemem no --sync outcore --sort src > results_impl2/"${file_name}"_1024_2_outcore_src.stat

./sssp --input test_graphs/"${file_name}".txt --bsize 256 --bcount 8 --output output.txt --method tpe --usemem no --sync outcore --sort dest >  results_impl2/"${file_name}"_256_8_outcore_dest.stat
./sssp --input test_graphs/"${file_name}".txt --bsize 384 --bcount 5 --output output.txt --method tpe --usemem no --sync outcore --sort dest >  results_impl2/"${file_name}"_384_5_outcore_dest.stat
./sssp --input test_graphs/"${file_name}".txt --bsize 512 --bcount 4 --output output.txt --method tpe --usemem no --sync outcore --sort dest > results_impl2/"${file_name}"_512_4_outcore_dest.stat
./sssp --input test_graphs/"${file_name}".txt --bsize 768 --bcount 2 --output output.txt --method tpe --usemem no --sync outcore --sort dest > results_impl2/"${file_name}"_768_2_outcore_dest.stat
./sssp --input test_graphs/"${file_name}".txt --bsize 1024 --bcount 2 --output output.txt --method tpe --usemem no --sync outcore --sort dest > results_impl2/"${file_name}"_1024_2_outcore_dest.stat

done
