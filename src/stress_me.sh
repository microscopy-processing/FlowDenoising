if test ! -f TS_026.mrc; then
    wget https://ftp.ebi.ac.uk/empiar/world_availability/10988/data/DEF/tomograms/TS_026.rec -O TS_026.mrc
fi
python flowdenoising.py -i TS_026.mrc -o /tmp/TS_026_SDPG.mrc -v 1
