cd ../python/data_ppl
python3 -u feateng.py $1
python3 -u get_dataset_sum.py $1

python3 -u insert_es.py $1
python3 -u pre_search.py $1 100 11 rim
