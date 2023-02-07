if test ! -f EMPIAR-10331_crop.mrc ; then
    FILEID="1jYL6FEMeWGXO0KYlCb9udrICc2qaZLHB"
    OUTPUT_FILENAME="small_vol.mrc"
    wget --no-check-certificate 'https://docs.google.com/uc?export=download&id={FILEID}' -O {OUTPUT_FILENAME}
fi
python flowdenoising.py -i small_vol.mrc -o small_vol_smoothened.mrc -m -v 1