# ./run_wse3.sh 64 4096 4096 4096 false | tee log/wse3_4k_64.log &
# ./run_wse3.sh 128 4096 4096 4096 false | tee log/wse3_4k_128.log &
# ./run_wse3.sh 256 4096 4096 4096 false | tee log/wse3_4k_256.log &
# ./run_wse3.sh 512 4096 4096 4096 false | tee log/wse3_4k_512.log &

# wait

# ./run_wse3.sh 64 8192 8192 8192 false | tee log/wse3_8k_64.log &
# ./run_wse3.sh 128 8192 8192 8192 false | tee log/wse3_8k_128.log &
# ./run_wse3.sh 256 8192 8192 8192 false | tee log/wse3_8k_256.log &
# ./run_wse3.sh 512 8192 8192 8192 false | tee log/wse3_8k_512.log &

# wait

# ./run_wse3.sh 128 16384 16384 16384 false | tee log/wse3_16k_128.log &
# ./run_wse3.sh 256 16384 16384 16384 false | tee log/wse3_16k_256.log &
# ./run_wse3.sh 512 16384 16384 16384 false | tee log/wse3_16k_512.log &

./run_wse3.sh 180 2160 2160 2160 false | tee log/wse3_2k_180.log &
./run_wse3.sh 360 2160 2160 2160 false | tee log/wse3_2k_360.log &
./run_wse3.sh 540 2160 2160 2160 false | tee log/wse3_2k_540.log &
./run_wse3.sh 720 2160 2160 2160 false | tee log/wse3_2k_720.log &
wait

./run_wse3.sh 180 4320 4320 4320 false | tee log/wse3_4k_180.log &
./run_wse3.sh 360 4320 4320 4320 false | tee log/wse3_4k_360.log &
./run_wse3.sh 540 4320 4320 4320 false | tee log/wse3_4k_540.log &
./run_wse3.sh 720 4320 4320 4320 false | tee log/wse3_4k_720.log &
wait

./run_wse3.sh 180 8640 8640 8640 false | tee log/wse3_8k_180.log &
./run_wse3.sh 360 8640 8640 8640 false | tee log/wse3_8k_360.log &
./run_wse3.sh 540 8640 8640 8640 false | tee log/wse3_8k_540.log &
./run_wse3.sh 720 8640 8640 8640 false | tee log/wse3_8k_720.log &
wait