
rm -rf results_impl1
mkdir results_impl1

files=('amazon0312' 'msdoor' 'roadNet-CA' 'soc-LiveJournal1' 'USA-road-d.CAL' 'web-Google')	
for file_name in "${files[@]}"
do
./sssp --input test_graphs/"${file_name}".txt --bsize 256 --bcount 8 --output output.txt --method bmf --usesmem no --sync incore --sort src >  results_impl1/"${file_name}"_256_8_bmf_incore_src.stat
./sssp --input test_graphs/"${file_name}".txt --bsize 384 --bcount 5 --output output.txt --method bmf --usesmem no --sync incore --sort src >  results_impl1/"${file_name}"_384_5_bmf_incore_src.stat
./sssp --input test_graphs/"${file_name}".txt --bsize 512 --bcount 4 --output output.txt --method bmf --usesmem no --sync incore --sort src > results_impl1/"${file_name}"_512_4_bmf_incore_src.stat
./sssp --input test_graphs/"${file_name}".txt --bsize 768 --bcount 2 --output output.txt --method bmf --usesmem no --sync incore --sort src > results_impl1/"${file_name}"_768_2_bmf_incore_src.stat
./sssp --input test_graphs/"${file_name}".txt --bsize 1024 --bcount 2 --output output.txt --method bmf --usesmem no --sync incore --sort src > results_impl1/"${file_name}"_1024_2_bmf_incore_src.stat


./sssp --input test_graphs/"${file_name}".txt --bsize 256 --bcount 8 --output output.txt --method bmf --usesmem no --sync incore --sort dest >  results_impl1/"${file_name}"_256_8_bmf_incore_dest.stat
./sssp --input test_graphs/"${file_name}".txt --bsize 384 --bcount 5 --output output.txt --method bmf --usesmem no --sync incore --sort dest >  results_impl1/"${file_name}"_384_5_bmf_incore_dest.stat
./sssp --input test_graphs/"${file_name}".txt --bsize 512 --bcount 4 --output output.txt --method bmf --usesmem no --sync incore --sort dest > results_impl1/"${file_name}"_512_4_bmf_incore_dest.stat
./sssp --input test_graphs/"${file_name}".txt --bsize 768 --bcount 2 --output output.txt --method bmf --usesmem no --sync incore --sort dest > results_impl1/"${file_name}"_768_2_bmf_incore_dest.stat
./sssp --input test_graphs/"${file_name}".txt --bsize 1024 --bcount 2 --output output.txt --method bmf --usesmem no --sync incore --sort dest > results_impl1/"${file_name}"_1024_2_bmf_incore_dest.stat


./sssp --input test_graphs/"${file_name}".txt --bsize 256 --bcount 8 --output output.txt --method bmf --usesmem no --sync outcore --sort src >  results_impl1/"${file_name}"_256_8_bmf_outcore_src.stat
./sssp --input test_graphs/"${file_name}".txt --bsize 384 --bcount 5 --output output.txt --method bmf --usesmem no --sync outcore --sort src >  results_impl1/"${file_name}"_384_5_bmf_outcore_src.stat
./sssp --input test_graphs/"${file_name}".txt --bsize 512 --bcount 4 --output output.txt --method bmf --usesmem no --sync outcore --sort src > results_impl1/"${file_name}"_512_4_bmf_outcore_src.stat
./sssp --input test_graphs/"${file_name}".txt --bsize 768 --bcount 2 --output output.txt --method bmf --usesmem no --sync outcore --sort src > results_impl1/"${file_name}"_768_2_bmf_outcore_src.stat
./sssp --input test_graphs/"${file_name}".txt --bsize 1024 --bcount 2 --output output.txt --method bmf --usesmem no --sync outcore --sort src > results_impl1/"${file_name}"_1024_2_bmf_outcore_src.stat

./sssp --input test_graphs/"${file_name}".txt --bsize 256 --bcount 8 --output output.txt --method bmf --usesmem no --sync outcore --sort dest >  results_impl1/"${file_name}"_256_8_bmf_outcore_dest.stat
./sssp --input test_graphs/"${file_name}".txt --bsize 384 --bcount 5 --output output.txt --method bmf --usesmem no --sync outcore --sort dest >  results_impl1/"${file_name}"_384_5_bmf_outcore_dest.stat
./sssp --input test_graphs/"${file_name}".txt --bsize 512 --bcount 4 --output output.txt --method bmf --usesmem no --sync outcore --sort dest > results_impl1/"${file_name}"_512_4_bmf_outcore_dest.stat
./sssp --input test_graphs/"${file_name}".txt --bsize 768 --bcount 2 --output output.txt --method bmf --usesmem no --sync outcore --sort dest > results_impl1/"${file_name}"_768_2_bmf_outcore_dest.stat
./sssp --input test_graphs/"${file_name}".txt --bsize 1024 --bcount 2 --output output.txt --method bmf --usesmem no --sync outcore --sort dest > results_impl1/"${file_name}"_1024_2_bmf_outcore_dest.stat

./sssp --input test_graphs/"${file_name}".txt --bsize 256 --bcount 8 --output output.txt --method bmf --usesmem yes --sync outcore --sort dest >  results_impl1/"${file_name}"_256_8_bmf_outcore_dest_shared.stat
./sssp --input test_graphs/"${file_name}".txt --bsize 384 --bcount 5 --output output.txt --method bmf --usesmem yes --sync outcore --sort dest >  results_impl1/"${file_name}"_384_5_bmf_outcore_dest_shared.stat
./sssp --input test_graphs/"${file_name}".txt --bsize 512 --bcount 4 --output output.txt --method bmf --usesmem yes --sync outcore --sort dest > results_impl1/"${file_name}"_512_4_bmf_outcore_dest_shared.stat
./sssp --input test_graphs/"${file_name}".txt --bsize 768 --bcount 2 --output output.txt --method bmf --usesmem yes --sync outcore --sort dest > results_impl1/"${file_name}"_768_2_bmf_outcore_dest_shared.stat
./sssp --input test_graphs/"${file_name}".txt --bsize 1024 --bcount 2 --output output.txt --method bmf --usesmem yes --sync outcore --sort dest > results_impl1/"${file_name}"_1024_2_bmf_outcore_dest_shared.stat
done
