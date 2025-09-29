#data
year=[1991,1992,1993,1994,1995]
quartely_data=[102,120,90,78,110,126,95,83,111,128,97,86,115,135,103,91,122,144,110,98]

#a -- obtain a 4-centered moving average
import prettytable as pt
import numpy as np
table2=pt.PrettyTable()

year_col=[]
for i in range(len(year)*4):
  if i%4==0:
    year_col.append(year[i//4])
  else:
    year_col.append('')
table2.add_column("Year",year_col)

quarter_col=["Spring","Summer","Fall","Winter"]*5
table2.add_column("Quarter",quarter_col)

quartely_data=np.array(quartely_data)
table2.add_column("Sales",quartely_data)

moving_total=[]
#calculate the 4 quarter moving total
moving_total.append('')
moving_total.append('')
for i in range(len(quartely_data)-2):
  moving_total.append(sum(quartely_data[i:i+4]))
moving_total[-1]=''
table2.add_column("4-Quarter Moving total",moving_total)

#calculate the 4 quarter moving total average
moving_total_average=[]
for i in range(len(moving_total)):
  if moving_total[i]=='':
    moving_total_average.append('')
  else:
    moving_total_average.append(moving_total[i]/4)
table2.add_column("4-Quarter Moving average",moving_total_average)

#calculate the 4 quarter centered moving average
centered_moving_average=[]
for i in range(len(moving_total_average)):
  if i<2 or i>len(moving_total_average)-3:
    centered_moving_average.append('')
  else:
    centered_moving_average.append(np.mean(moving_total_average[i:i+2]))
table2.add_column("4-Quarter Centered Moving average",centered_moving_average)

#b -- calculate the ratio of actual to moving average
ratio=[]
for i in range(len(quartely_data)):
  if centered_moving_average[i]=='':
    ratio.append('')
  else:
    ratio.append((quartely_data[i]/centered_moving_average[i])*100)
table2.add_column("Ratio",ratio)


print(table2)

#c -- calculate the modified seasonal indices and seasonal indices

table2a=pt.PrettyTable()
table2a.add_column("Year",year)
arranged_ratio=[]
for i in range(0,len(ratio),4):
  arranged_ratio.append(ratio[i:i+4])
quarters=[]
for i in range(len(arranged_ratio[0])):
  t=[]
  for j in range(len(arranged_ratio)):
    if arranged_ratio[j][i]=='':
      t.append('')
    else:
      t.append(float(arranged_ratio[j][i]))
  quarters.append(t)
for q in quarters:
  table2a.add_column("Q"+str(quarters.index(q)+1),q)

print(table2a)

modified_means = []
for quarter_data in quarters:
    # Filter out empty strings
    valid_data = [x for x in quarter_data if x != '']
    if len(valid_data) >= 3: # Need at least 3 values to remove max and min
        valid_data.sort()
        modified_mean = np.mean(valid_data[1:-1]) # Remove min and max and calculate mean
        modified_means.append(modified_mean)
    else:
        modified_means.append(np.mean(valid_data) if valid_data else '')

print("\nModified Means for each quarter:")
for i, mean in enumerate(modified_means):
    print(f"Q{i+1}: {mean:.2f}")


sum_modified_means=sum(modified_means)
print("\nSum of modified means:",sum_modified_means)
sum_desired=400
print("Sum of desired means:",sum_desired)
adjusting_constant=sum_desired/sum_modified_means
print("Adjusting constant:",adjusting_constant)

normalized_means = [mean * adjusting_constant for mean in modified_means]

print("\nNormalized Means (Seasonal Indices):")
for i, norm_mean in enumerate(normalized_means):
    print(f"Q{i+1}: {norm_mean:.2f}")

sum_normalized_means=sum(normalized_means)
print("\nSum of normalized means:",sum_normalized_means)


# additional -- deseasonalizing the data
deseasonalized_data = []
for i in range(len(quartely_data)):
    quarter_index = i % 4
    deseasonalized_value = (quartely_data[i] / normalized_means[quarter_index]) * 100
    deseasonalized_data.append(deseasonalized_value)

table3=pt.PrettyTable()
table3.add_column("Year",year_col)
table3.add_column("Quarter",quarter_col)
table3.add_column("Sales",quartely_data)
table3.add_column("Deseasonalized Data", deseasonalized_data)
print(table3)

import matplotlib.pyplot as plt
quarter_labels = [f"{y} Q{q+1}" for y in year for q in range(4)]
plt.figure(figsize=(12, 6))
plt.plot(quarter_labels, deseasonalized_data, marker='o', linestyle='-', label='Deseasonalized Data')
plt.plot(quarter_labels, quartely_data, marker='x', linestyle='--', label='Original Sales Data')
plt.xticks(rotation=90)
plt.xlabel("Year and Quarter")
plt.ylabel("Sales/Deseasonalized Sales")
plt.title("Original and Deseasonalized Sales Data")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()