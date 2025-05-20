#!/bin/bash

rm *clean.txt
rm all_data.txt

for infile in *perf*; do
  outfile=${infile%.*}_clean.txt
  echo "infile: ${infile}, outfile: ${outfile}"

  rm no_text_tmp.txt
  rm no_comma_tmp.txt
  rm all_one_line_tmp_unsorted.txt

  # first things first, remove commas in number strings, and remove text
  awk '(NR>2) { if ($1 != "#" ) { print $1,$2,$3,$4 }}' ${infile} >> no_text_tmp.txt
  perl -pe 's/(?<=\d),(?=\d)//g' no_text_tmp.txt >> no_comma_tmp.txt

  # now get all data on the same line given the same timestamp and same cpu
  awk ' BEGIN {prev_insn_sum[$2] = 0}
  	$4 == "instructions:u" {
  		insn[$1][$2] = $3;
		insn_sum[$1][$2] = prev_insn_sum[$2] + $3;
		prev_insn_sum[$2] = insn_sum[$1][$2];
	}
	$4 == "LLC-loads" {
  		loads[$1][$2] = $3;
	}
	$4 == "LLC-stores" {
		stores[$1][$2] = $3;
	}
	$4 == "LLC-loads-misses" {
		misses[$1][$2] = $3;
	}
	END {
	    for (i in insn)
		    for (j in insn[i])
		    	print i, j, insn_sum[i][j], insn[i][j], loads[i][j], stores[i][j], misses[i][j]

   	}' no_comma_tmp.txt >> all_one_line_tmp_unsorted.txt
  sort -n all_one_line_tmp_unsorted.txt > ${outfile}
  cat ${outfile} >> all_data.txt
  sync

done

# I need to gather all the data from here that belongs to each unique cache and membw file
# now generate files that are resource seperate but group indexes
cache=1
#for ((cache = 1 ; cache <= 1048575 ; cache = cache + cache + 1)); do
  #for ((membw = 72 ; membw <= 1440 ; membw = membw + 72)); do
    rm tmp.txt
    for infile1 in *${cache}_${membw}_perf_mt_*_clean.txt; do
      # Use awk to find the position of the 4th occurrence of '_'
      position=$(echo "${infile1}" | awk -F '_' '{print index($0, $5)}')
      # Use cut to extract the substring up to the 4th occurrence of '_'
      outfile=$(echo "${infile1}" | cut -c 1-$position)t_allidx_clean.txt
      echo "infiles: ${infile1}, outfile: ${outfile}"
      cat ${infile1} >> tmp.txt
    done
    cp tmp.txt ${outfile}
  #done
#done
