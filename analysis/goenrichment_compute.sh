#!/bin/bash
#SBATCH --job-name=goenrich
#SBATCH --output=goenrich.txt

for num in {0..72}  
do  
python3 scripts/convert_symbol_to_entrez.py --id $num
python3 scripts/find_enrichment.py ./data/study_geneids.txt ./data/pop_geneids.txt gene2go --method=fdr_bh --outfile "../goatools_result_new/output_"$num$".csv"
done  
