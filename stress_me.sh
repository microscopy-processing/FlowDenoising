if test ! -f TS_026.mrc; then
    wget https://ftp.ebi.ac.uk/empiar/world_availability/10988/data/DEF/tomograms/TS_026.rec TS_026.mrc
fi
python flowdenoising.py -i TS_026.mrc -o filtered_stack.mrc -m -v 1
