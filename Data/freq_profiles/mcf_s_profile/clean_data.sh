#!/bin/bash

rm *clean.txt

for infile in *perf*; do
  outfile=${infile%.*}_clean.txt
  echo "infile: ${infile}, outfile: ${outfile}"

  rm no_text_tmp.txt
  rm no_comma_tmp.txt
  rm all_one_line_tmp_unsorted.txt
  rm all_data.txt

  # first things first, remove commas in number strings, and remove text
  awk '(NR>2) { if ($1 != "#" ) { print $1,$2,$3 }}' ${infile} >> no_text_tmp.txt
  perl -pe 's/(?<=\d),(?=\d)//g' no_text_tmp.txt >> no_comma_tmp.txt

  # now get all data on the same line given the same timestamp
  awk ' BEGIN {prev_insn_sum = 0}
  	$3 == "instructions:u" {
		echo "hi"
  		insn[$1] = $2;
		insn_sum[$1] = prev_insn_sum + $2;
		prev_insn_sum = insn_sum[$1];
	}
	$3 == "LLC-loads" {
  		loads[$1] = $2;
	}
	$3 == "LLC-loads-misses" {
		misses[$1] = $2;
	}
	END {
	    for (i in insn)
		    print i, insn_sum[i], insn[i], loads[i], stores[i], misses[i]

   	}' no_comma_tmp.txt >> all_one_line_tmp_unsorted.txt
  sort -n all_one_line_tmp_unsorted.txt > ${outfile}
  cat ${outfile} >> all_data.txt
done

# now generate the are resource seperate but group indexes
for infile1 in *perf_1_clean.txt; do
  infile2=${infile1%1*}2_clean.txt
  infile3=${infile1%1*}3_clean.txt
  outfile=${infile1%1*}allidx_clean.txt
  echo "infiles: ${infile1}, ${infile2}, ${infile3}, outfile: ${outfile}"
  rm tmp.txt
  cat ${infile1} >> tmp.txt
  cat ${infile2} >> tmp.txt
  cat ${infile3} >> tmp.txt
  #sort -k 2 -n tmp.txt > ${outfile} sort based on sum of insn
  sort -k 1 -n tmp.txt > ${outfile} # sort based on time
done

#	$3 == "LLC-stores" {
#		stores[$1] = $2;
#	}
