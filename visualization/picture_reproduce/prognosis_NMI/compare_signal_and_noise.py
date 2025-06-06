import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from Tool_Functions.load_in_data import read_in_text

line_list = read_in_text('/home/zhoul0a/Desktop/prognosis_project/threshold.txt')
value_list = []
for line in line_list:
    value_list.append(float(line.split(' ')[3][1:-1]))

value_list = np.array(value_list)
value_list = value_list - np.mean(value_list)

print(np.std(value_list))

median_lesion_severity_list = []
data_list = read_in_text('/home/zhoul0a/Desktop/prognosis_project/transfer/median_lesion_severity')

for data in data_list:
    median_lesion_severity_list.append(float(data[:-1]))
print(np.std(median_lesion_severity_list))
print(np.mean(median_lesion_severity_list))

print(np.std(value_list) / np.std(median_lesion_severity_list))

x_l = []

y_l = []

a = median_lesion_severity_list
b = value_list

x_t = []
for y in a:
    y_l.append(y)
    x_t.append('Median Lesion Severity')
for y in b:
    y_l.append(y)
    x_t.append('Scan-level Bias for Baseline')

ax = sns.boxplot(x=x_t, y=y_l, order=['Median Lesion Severity', 'Scan-level Bias for Baseline'])
sns.swarmplot(x=x_t, y=y_l, order=['Median Lesion Severity', 'Scan-level Bias for Baseline'], color=".25")
# ax.set_ylim([min(y_l + y_l_2), max(y_l + y_l_2)])
ax.set_ylabel('Hounsfield Unit (HU)')
ax.set_title('Comparision between Information and Noise')

plt.rcParams['savefig.dpi'] = 600
plt.rcParams['figure.dpi'] = 600
plt.show()
exit()
plt.savefig('/home/zhoul0a/Desktop/Signal_and_Bias.svg')
plt.close()

