import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

################################
#### Check training convergence
################################
# Load training mae trend - train_mae[0] is the mae for the first fold
train_mae = np.load("folds_mae_train.npy")
N = 5  # N times running mean
col = 5
row = 2
fold_num = 10
# Visualization of each filter of the layer
fig = plt.figure(figsize=(col*4, row*2))
for i in range(fold_num):
    ax = fig.add_subplot(row, col, i + 1)
    train_run_mean = np.convolve(train_mae[i], np.ones((N,)) / N, mode='valid') / 3600.
    ax.plot(train_run_mean)
fig.subplots_adjust(wspace=.3, hspace=.3)
plt.show()

# Load the file containing fold_corrcoef, fold_mae_test, fold_predicted_seconds and fold_test_Y for the k folds
results = np.load("cme_results.npz")
lst = results.files
print(lst)
cc_ave = results[lst[0]]
mae_ave = results[lst[1]]
pred = results[lst[2]] / 3600.
pred = np.concatenate(pred)
true = results[lst[3]] / 3600.
true = np.concatenate(true)
error = pred - true

##################
#### Task 1
##################
# Calculate 13 maes for the 13 bins [10,150,10], e.g. [10,20] hrs... [130-140] hrs etc.
ranges_mae = []
for n_bin in range(13):
    range_pred = []
    range_true = []
    for ind, value in enumerate(true):
        if (10+10*n_bin) <= true[ind] <= (20+10*n_bin):
            range_pred.append(pred[ind])
            range_true.append(true[ind])
    range_pred = np.array(range_pred)
    range_true = np.array(range_true)
    print("range_pred: ", range_pred)
    print("range_true: ", range_true)
    print("range_true shape: ", range_true.shape)
    ranges_mae.append(np.mean(abs(range_pred-range_true)))
print("ranges_mae: ", ranges_mae)

# Replace nan in ranges_mae by the mean of its previous and next mae values for later plotting
for i in range(len(ranges_mae)):
    if np.isnan(ranges_mae[i]):
        ranges_mae[i] = np.mean([ranges_mae[i-1], ranges_mae[i+1]])
print("ranges_mae: ", ranges_mae)

# Hist of true and their corresponding maes
# left-axis
fig, ax1 = plt.subplots()
n, bins, patches = ax1.hist(true, bins=range(10, 150, 10), fill=False, linewidth=1, ec='b')
print("image number in bins: ", n)
ax1.set_xlabel('CME transit times (hrs)')
# Make the y-axis label, ticks and tick labels match the line color.
ax1.set_ylabel('Number of images', color='b')
ax1.tick_params('y', colors='b')
# right-axis
ax2 = ax1.twinx()
x_ranges = np.arange(15, 145, 10)
ax2.plot(x_ranges, ranges_mae, 'r')
ax2.set_ylabel('MAEs (hrs)', color='r')
ax2.tick_params('y', colors='r')
# fig property
fig.tight_layout()
plt.show()

##################
#### Task 2
##################
# Hist of predicted
good_pred = np.where(abs(error) <= 12)
print("good_pred", len(good_pred[0]))
plt.hist(error, bins=range(0, 44, 4), fill=False, linewidth=1)
axes = plt.gca()
axes.set_xlim(0, 41)
axes.set_xticks(np.arange(0, 44, 4))
axes.set_xticklabels(np.arange(0, 44, 4))
plt.xlabel('MAEs (hrs)')
plt.ylabel('Number of images')
plt.show()

##################
#### Task 3
##################
# Scatter plot of time interval of arrival
fig = plt.figure()
plt.scatter(true, pred, s=2, color='k')
plt.plot([10, 140], [10, 140], 'b--', linewidth=0.9)
plt.plot([25, 140], [10, 125], 'r--', linewidth=0.9)
plt.plot([10, 125], [25, 140], 'r--', linewidth=0.9)
axes = plt.gca()
axes.set_xlim([10, 140])
axes.set_ylim([10, 140])
plt.xlabel('Actual transit time (hrs)')
plt.ylabel('Predicted transit time (hrs)')
plt.show()
plt.close(fig)

##################
#### Task 4
##################
# Zip testing files with their errors, predicted values and true values
# Load testing image names for all the 1122 testing data
test_img_names = np.load('shuffled_img_names.npy')
print(test_img_names.shape)
np.savez('test_errors.npz', name1=test_img_names, name2=error, name3=pred, name4=true)

