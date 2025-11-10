rm -rf STACKS MWCS DTT db.ini msnoise.sqlite

msnoise db init --tech 1

msnoise config set startdate=2016-04-01
msnoise config set enddate=2016-05-16
msnoise config set components_to_compute=22
# msnoise config set components_to_compute_single_station=11
# msnoise config set remove_response='Y'
# msnoise config set response_format=inventory
# msnoise config set response_prefilt=0.01,0.05,99,99.5

msnoise config set preprocess_lowpass=99.5
msnoise config set preprocess_highpass=0.1
msnoise config set cc_sampling_rate=2000
msnoise config set preprocess_max_gap=3600

msnoise config set mov_stack=1
msnoise config set stack_method='linear'

msnoise config set data_structure=custom.py
msnoise config set data_folder=.
msnoise populate


msnoise config set maxlag=3
msnoise config set overlap=0.5
# msnoise config set keep_days=Y
# msnoise config set analysis_duration=86400  # 1 day
msnoise config set corr_duration=1800  # 4h
msnoise config set hpc='Y'

msnoise db execute 'insert into filters (ref, low, mwcs_low, high, mwcs_high, rms_threshold, mwcs_wlen, mwcs_step, used) values (1, 1, 1, 99, 99, 0.0, 12.0, 4.0, 1)'
# msnoise db execute 'insert into filters (ref, low, mwcs_low, high, mwcs_high, rms_threshold, mwcs_wlen, mwcs_step, used) values (2, 20, 20, 30, 30, 0.0, 12.0, 4.0, 1)'
# msnoise db execute 'insert into filters (ref, low, mwcs_low, high, mwcs_high, rms_threshold, mwcs_wlen, mwcs_step, used) values (3, 30, 30, 40, 40, 0.0, 12.0, 4.0, 1)'
# msnoise db execute 'insert into filters (ref, low, mwcs_low, high, mwcs_high, rms_threshold, mwcs_wlen, mwcs_step, used) values (4, 40, 40, 50, 50, 0.0, 12.0, 4.0, 1)'
# msnoise db execute 'insert into filters (ref, low, mwcs_low, high, mwcs_high, rms_threshold, mwcs_wlen, mwcs_step, used) values (5, 50, 50, 60, 60, 0.0, 12.0, 4.0, 1)'
# msnoise db execute 'insert into filters (ref, low, mwcs_low, high, mwcs_high, rms_threshold, mwcs_wlen, mwcs_step, used) values (6, 60, 60, 70, 70, 0.0, 12.0, 4.0, 1)'
# msnoise db execute 'insert into filters (ref, low, mwcs_low, high, mwcs_high, rms_threshold, mwcs_wlen, mwcs_step, used) values (7, 70, 70, 80, 80, 0.0, 12.0, 4.0, 1)'
# msnoise db execute 'insert into filters (ref, low, mwcs_low, high, mwcs_high, rms_threshold, mwcs_wlen, mwcs_step, used) values (8, 80, 80, 90, 90, 0.0, 12.0, 4.0, 1)'

msnoise scan_archive --path ./forest/DP2 --recursively --init
#msnoise populate --fromDA
msnoise new_jobs

msnoise -t 2 compute_cc

msnoise reset -a STACK
msnoise new_jobs --hpc CC:STACK
msnoise stack -r
msnoise reset STACK
msnoise stack -m
