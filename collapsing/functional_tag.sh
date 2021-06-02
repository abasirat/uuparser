
ud_path='/home/staff/abasirat/Lab/Universal_Dependency_Parsing/ud/ud-treebanks-v2.3/'
ud_path_out='/home/staff/abasirat/Lab/Universal_Dependency_Parsing/ud/ud-treebanks-v2.3-functionaltag/'

if [ ! -d $ud_path_out ]
then
  mkdir -p $ud_path_out
fi

files=$(find ${ud_path} -regextype sed -regex ".*.conllu")
for file in $files
do
  fname=$(basename $file)
  dname=$(dirname $file)
  corpus=$(basename $dname)
  
  out_dir=${ud_path_out}/${corpus}
  if [ ! -d $out_dir ]
  then
    mkdir -p $out_dir
  fi
  out_file=${out_dir}/${fname}
  echo $file
  echo $out_file

  cat $file | awk -F'\t' '{
    if ((NF == 10) && ($1 ~ /^[0-9].*/)){ 
      if ($8 ~ /aux|det|case|cc|clf|cop|mark/) {
        #$2=sprintf("%s__functional",$2)
        $4 = "FUNCTIONAL"
      }
      else {
        $4 = "CONTENT"
      }
      #for (i=1;i<10;i++) {
      #  printf("%s\t",$i)
      #}
      #printf("%s\n",$10)
      printf("%s\t%s\n",$2,$4)
    }
    else{
        if ( $0 !~ /^#/){
          print($0)
        }
      }
    }' > ${out_file}
done

