# CCT-LSTM_reproduce

## integrity_and_masterManifest.py 
### Check the integrity of UBFC-Phys dataset by the following logic:
1. In selected folder, if "UBFC-Phys" folder exists, then it will be used as the dataset.
2. In "UBFC-Phys", if "Data" folder exists.
3. If there are 56 folders named "s1(subject number)", "s2", ..., "s56" in "Data" folder.
4. For each subject folder:
   1. if there are 3 avi files named "vid_s1(subject number)_1.avi", "vid_s1_2.avi", "vid_s1_3.avi".
   2. if there are 3 csv files named "bvp_s1(subject number)_T1.csv", "bvp_s1_T2.csv", "bvp_s1_T3.csv".
   3. if there are 3 csv files named "eda_s1(subject number)_T1.csv", "eda_s1_T2.csv", "eda_s1_T3.csv".
   4. if there is 1 txt file named "info_s1(subject number).txt".
   5. if there is 1 csv file named "selfReportedAnx_s1(subject number).csv".
If any of the above conditions are not met, it will print an error message and exit.

### After integrity check, it will generate a master manifest file named "masterManifest.csv" in the selected folder.
The master manifest file will contain the following columns:
- subject: subject number
- group: 'test' or 'ctrl'