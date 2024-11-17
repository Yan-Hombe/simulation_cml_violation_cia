##############################################################################
# visualization of simulation results
# this code is used to create visualizations of the simulation results
# input: csv files from simulation
# output: csv file with all results of the simulation
##############################################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
import os
import scienceplots  # Importing this enables the science style
import itertools


##############################################################################
# analysis

# get path
wd = os.getcwd()
source_path = r"C:\Users\yh-95\OneDrive - Universität St.Gallen\python_env\envmaster\Master_Thesis"

path = source_path + '/simulations/scenariosDGP/aggregated' # path for artificial DGP

csv_list = os.listdir(path)

# Assuming 'path' and 'csv_list' are defined
df_csv = pd.read_csv(path + '/' + csv_list[0])

for file in csv_list[1:]:
    print(file)
    new_file = pd.read_csv(path + '/' + file)
    df_csv = pd.concat([df_csv, new_file], ignore_index=True) 


ml_algo_order = ['Linear Regression', 'DML', 'T-learner', 'X-learner', 'XBCF']

# Convert the ml_algo column to a categorical type with the specified order
df_csv['ml_algo'] = pd.Categorical(df_csv['ml_algo'], categories=ml_algo_order, ordered=True)

# drop any duplicates 
index_drop_na_dupl = df_csv[['n_observations', 'gamma', 'lam', 'ml_algo', 'linear_di', 'linear','share_omit', ]].drop_duplicates().dropna().index


df_scenario = pd.read_csv(source_path +'/simulations/correlation/df_scenarios.csv', header= 0)

df_csv = df_csv.merge(df_scenario, how = 'left', on = ['share_omit', 'gamma', 'lam', 'linear'])




# some manipulation of the data
def manipulate_data(df):
    df['RMSE'] = np.sqrt(df['MSE'])
    df['DGP'] = np.where(df['linear'] == True, 'Linear' ,'Non-Linear')
    return df

df_csv = manipulate_data(df_csv)


# df for special DML results
path = source_path + '/simulations/scenariosDGP/DML_30000'

csv_list = os.listdir(path)

# Assuming 'path' and 'csv_list' are defined
df_DML = pd.read_csv(path + '/' + csv_list[0])

df_DML = df_DML.merge(df_scenario, how = 'left', on = ['share_omit', 'gamma', 'lam', 'linear'])

df_DML = manipulate_data(df_DML)


# DML different propensity score
path = source_path + '/simulations/scenariosDGP/DML_diff_propensity' # path for artificial DGP

csv_list = os.listdir(path)

# Assuming 'path' and 'csv_list' are defined
df_DML_prop = pd.read_csv(path + '/' + csv_list[0])

for file in csv_list[1:]:
    print(file)
    new_file = pd.read_csv(path + '/' + file)
    df_DML_prop = pd.concat([df_DML_prop, new_file], ignore_index=True) 

df_DML_prop = df_DML_prop.merge(df_scenario, how = 'left', on = ['share_omit', 'gamma', 'lam', 'linear'])

df_DML_prop = manipulate_data(df_DML_prop)




df = df_csv[df_csv['linear'] == True]
# create a plot for plotting Correlation against absolute bias (abs_bias)
# Assuming df_csv has columns 'Correlation' and 'abs_bias'

# Use the 'science' style
plt.style.use(['science', 'no-latex'])

# Directly set the font size for different elements
plt.rcParams.update({
    'font.size': 14,          # Base font size for all elements
    'axes.titlesize': 16,     # Title font size
    'axes.labelsize': 14,     # Axis label font size
    'xtick.labelsize': 14,    # X-tick label size
    'ytick.labelsize': 14,    # Y-tick label size
    'legend.fontsize': 14,    # Legend font size
    'figure.titlesize': 16    # Overall figure title font size
})


# Create the lmplot without the default legend
lm = sns.lmplot(
    data=df, x='Correlation', y='bias', hue='ml_algo', palette='deep',
    height=6, aspect=1.5, markers='o', scatter_kws={'s': 50}, legend=False
)

# Get the figure object from the FacetGrid (lmplot creates a FacetGrid object)
fig = lm.fig

# Get handles and labels for the custom legend
handles, labels = lm.ax.get_legend_handles_labels()

# Add a custom legend above the plot, spread across 5 columns
legend = fig.legend(
    handles=handles,
    labels=labels,
    title='ML Algorithm',  # Custom legend title
    loc='upper center',
    bbox_to_anchor=(0.5, 1.03),  # Adjust the vertical position to ensure title is visible
    fancybox=True,
    shadow=True,
    ncol=5  # Spread legend into 5 columns
)

# Set the legend title font size (specific to legend)
legend.get_title().set_fontsize(14)  # Ensure the title matches other fonts

# Adjust layout to make space for the legend and ensure it does not overlap with the plot
plt.tight_layout(rect=[0, 0, 1, 0.95])

# Set plot title and labels
plt.title('')
plt.xlabel('Correlation')
plt.ylabel('Bias')

# Show the plot
plt.show()

#lm.savefig(f'visuals/absoluteBiasandCorrelation.pdf', format='pdf', bbox_inches='tight')

# Show the plot
plt.show()

##############################################################################################################################################################
# tabel main results


def grouping_mean_df(df, group_by, mean = True):
    if mean == True:
        return df.groupby(group_by).mean().reset_index()
    else:
        return df.groupby(group_by).sum().reset_index()

def table_result_main(df, measures=['Bias'], scenario_order=None, multiple_measures=False):
    # Rename columns for clarity
    df = df.rename(columns={'ml_algo': 'Method', 'abs_bias': 'Abs Bias', 'n_observations': 'N Obs', 'bias':'Bias'})
    # Create the pivot table with 'linear' and 'Method' as rows and 'Scenario' as columns
    pivot_table = df.pivot_table(index=['N Obs', 'Method'], columns='Scenario', values=measures)

    if multiple_measures:

        df['ScenarioX'] = df.apply(lambda row: row['Scenario'] + '_' + row['x_omit'] 
                            if row['Scenario'] in ['S', 'W'] else row['Scenario'], axis=1)

        # Create the pivot table with 'linear' and 'Method' as rows and 'Scenario' as columns
        pivot_table = df.pivot_table(index=['N Obs', 'Method'], columns='ScenarioX', values=measures)

        # Swap the levels so 'Scenario' is the top-level and 'bias', 'Abs Bias' are next
        pivot_table = pivot_table.swaplevel(0, 1, axis=1)

        # Sort the columns to have bias and Abs Bias next to each other for each Scenario
        pivot_table = pivot_table.sort_index(axis=1, level=0)



    if not multiple_measures:

        # Calculate the total for each group (linear, Method)
        a = grouping_mean_df(df, ['Method', 'x_omit', 'N Obs'], mean=False)
        #total_measure = a.groupby(['N Obs', 'Method']).sum()[[measures[0]]]  # Assuming single measure for simplicity
        total_measure = a.groupby(['N Obs', 'Method'])[measures[0]].apply(lambda x: x.abs().sum()).to_frame()
        # Add the 'Total' column to the pivot table
        pivot_table[('Total', '')] = total_measure  # Empty level for 'Scenario' in total column

       # Reorder the Scenario columns if a specific order is provided
    if scenario_order is not None:
        if multiple_measures:

            ordered_columns = list(itertools.product(scenario_order, measures))

            # Reindex the pivot table based on the ordered columns
            pivot_table = pivot_table.reindex(columns=ordered_columns)


        else:

            ordered_columns = list(itertools.product(measures, scenario_order))

            pivot_table = pivot_table.reindex(columns=ordered_columns)
        
    # Round the values to 3 decimals
    pivot_table = pivot_table.round(3)

    return pivot_table


it_list = [[0.5, 0.5],
           [1.2, 0.5],
           [0.1, 0.1]]

DGP_name = 'Non-Linear'
measures = ['Bias', 'Abs Bias', 'RMSE']
measure = ['Bias', 'Abs Bias', 'RMSE']###########
multiple_measures = True
label = f"appendix{DGP_name}Result1"
caption = f"Appendix Result 1: Bias, Absolute Bias, and RMSE for {DGP_name} DGP."

x_omit = ['X999', 'X1', 'X2', 'X3'] ['X17', 'X18', 'X2 X18', 'X1 X2'] ['X17 X18', 'X8-X9 X11-X18', 'X19 X1-X4 X14-X18'] ['X19 X1-X9', 'X1-X2 X4-X18 X20']
strengthScenario = ['No Omit', 'S', 'S', 'S'] ['W' 'W', 'S W', 'S S'] ['W W' '10W', '5S 5W']  ['10S','All but 2S']


scenario = ['X999', 'X1', 'X2', 'X3']
scenario_order = ['No Omit', 'S_X1', 'S_X2', 'S_X3']

def main_result_latex(df_csv, scenario:list, scenario_order:list, it_list:list, measure:str, DGP_name:str, multiple_measures = False, label = None, caption = None):

    spec_pivot_tables = []

    for spec in it_list:

        gamma_value = spec[0]
        lambda_value = spec[1]
        
        df = df_csv[(df_csv['gamma'] == gamma_value) & (df_csv['lam'] == lambda_value) & (df_csv['DGP'] == DGP_name)] # lambda, gamma, and linear
        scenario = scenario
        scenario_order = scenario_order # always add an empty string at the end for Total in pivot table
        df = df[df['x_omit'].isin(scenario)]
        
        # Get the pivot table
        pivot_table = table_result_main(df, measures=measure, scenario_order = scenario_order, multiple_measures=multiple_measures)


        if label is None:
            name = DGP_name
        else:
            name = label

        if caption is None:
            caption = f'Result: Bias for {name} DGP.'


        # Convert pivot table to LaTeX format
        latex_table = pivot_table.to_latex(multicolumn=True, multirow=True, label= f'tab:{name}', caption=caption
        )

        insert_line =  f"\\hdashline \\multicolumn{{5}}{{l}}{{specification: lambda = {lambda_value} gamma = {gamma_value}}} \\\\\n"

        # Insert the new line before '\\bottomrule'
        latex_table_modified = latex_table.replace('\\bottomrule', insert_line + '\\bottomrule')

        spec_pivot_tables.append(latex_table_modified)

    if multiple_measures:
        label_index = spec_pivot_tables[0].find('\\label')

        # Find the position of the first '\midrule'
        buttomrule_index = spec_pivot_tables[0].find('\\bottomrule')

        latex_table_first = spec_pivot_tables[0][:buttomrule_index]
        to_label = spec_pivot_tables[0][:label_index] + f'\\begin{{adjustbox}}{{angle=90,center}}\n'
        from_label = spec_pivot_tables[0][label_index:buttomrule_index] 

        latex_table_first = to_label + from_label

        # Find the position of the first '\midrule'
        midrule_index = spec_pivot_tables[1].find('\\midrule')
        buttomrule_index = spec_pivot_tables[1].find('\\bottomrule')

        latex_table_middle = spec_pivot_tables[1][midrule_index:buttomrule_index]


        midrule_index = spec_pivot_tables[2].find('\\midrule')
        table_index = spec_pivot_tables[2].find('\\end{table}')
        # Keep everything from the first '\midrule' onward
        latex_table_last = spec_pivot_tables[2][midrule_index:table_index]

        latex_table_modified = latex_table_first + latex_table_middle + latex_table_last + f'\\end{{adjustbox}}\n\\end{{table}}'


    else:
        # Find the position of the first '\midrule'
        buttomrule_index = spec_pivot_tables[0].find('\\bottomrule')

        latex_table_first = spec_pivot_tables[0][:buttomrule_index]

        # Find the position of the first '\midrule'
        midrule_index = spec_pivot_tables[1].find('\\midrule')
        buttomrule_index = spec_pivot_tables[1].find('\\bottomrule')

        latex_table_middle = spec_pivot_tables[1][midrule_index:buttomrule_index]


        midrule_index = spec_pivot_tables[2].find('\\midrule')
        # Keep everything from the first '\midrule' onward
        latex_table_last = spec_pivot_tables[2][midrule_index:]

        # Combine the first part, custom line, and last part
        latex_table_modified = latex_table_first + latex_table_middle + latex_table_last

    return latex_table_modified

# Main result in result section: Linear
# gamma, lam
it_list = [[0.5, 0.1],
           [0.5, 0.5],
           [0.1, 0.1]]

DGP_name = 'Linear'
measures = ['Bias']


scenario = ['X999', 'X1', 'X15', 'X10-X19', 'X3-X20']
scenario_order = ['No Omit', 'S', 'W', '10W', 'All but 2S', ''] # always add an empty string at the end for Total in pivot table

latex_table_modified = main_result_latex(df_csv, scenario, scenario_order, it_list, measures, DGP_name)


# Write to LaTeX file
with open(f"tables/MainResultLinear.tex", "w") as f:
    f.write(latex_table_modified)

# Main result in result section: Non-Linear
it_list = [[0.5, 0.5],
           [0.5, 1.2],
           [0.1, 0.1]]

DGP_name = 'Non-Linear'
measure = ['Bias']


scenario = ['X999', 'X3', 'X17', 'X8-X9 X11-X18', 'X1-X2 X4-X18 X20']
scenario_order = ['No Omit', 'S', 'W', '10W', 'All but 2S', ''] # always add an empty string at the end for Total in pivot table

latex_table_modified = main_result_latex(df_csv, scenario, scenario_order, it_list, measure, DGP_name)

# Write to LaTeX file
with open(f"tables/MainResultNonLinear.tex", "w") as f:
    f.write(latex_table_modified)


## tables for appendix 
# linear
# gamma, lam
it_list = [[0.5, 0.1],
           [0.5, 0.5],
           [0.1, 0.1]]

DGP_name = 'Linear'
measures = ['Bias', 'Abs Bias', 'RMSE']
multiple_measures = True
label = f"appendix{DGP_name}Result1"
caption = f"Appendix Result 1: Bias, Absolute Bias, and RMSE for {DGP_name} DGP."

scenario = ['X999', 'X1', 'X2', 'X15']
scenario_order = ['No Omit', 'S_X1', 'S_X2', 'W_X15']

latex_table_modified = main_result_latex(df_csv, scenario, scenario_order, it_list, measures, DGP_name,  multiple_measures, label, caption)

# Write to LaTeX file
with open(f"tables/AppendixMainResult{DGP_name}_1.tex", "w") as f:
    f.write(latex_table_modified)


scenario = [ 'X16', 'X1 X16', 'X1 X2', 'X15 X16']
scenario_order = ['W_X16', 'S W', 'S S', 'W W'] 
label = f"appendix{DGP_name}Result2"
caption = f"Appendix Result 2: Bias, Absolute Bias, and RMSE for {DGP_name} DGP."

latex_table_modified = main_result_latex(df_csv, scenario, scenario_order, it_list, measures, DGP_name,  multiple_measures, label, caption)

# Write to LaTeX file
with open(f"tables/AppendixMainResult{DGP_name}_2.tex", "w") as f:
    f.write(latex_table_modified)

scenario = ['X10-X19', 'X1-X5 X15-X19', 'X1-X10', 'X3-X20']
scenario_order = ['10W', '5S 5W', '10S', 'All but 2S']
label = f"appendix{DGP_name}Result3"
caption = f"Appendix Result 3: Bias, Absolute Bias, and RMSE for {DGP_name} DGP."

latex_table_modified = main_result_latex(df_csv, scenario, scenario_order, it_list, measures, DGP_name,  multiple_measures, label, caption)

# Write to LaTeX file
with open(f"tables/AppendixMainResult{DGP_name}_3.tex", "w") as f:
    f.write(latex_table_modified)



# non linear
# gamma, lam
it_list = [[0.5, 0.5],
           [0.5, 1.2],
           [0.1, 0.1]]

DGP_name = 'Non-Linear'
measures = ['Bias', 'Abs Bias', 'RMSE']
multiple_measures = True
label = f"appendix{DGP_name}Result1"
caption = f"Appendix Result 1: Bias, Absolute Bias, and RMSE for {DGP_name} DGP."

x_omit = ['X999', 'X1', 'X2', 'X3', 'X17', 'X18', 'X2 X18', 'X1 X2','X17 X18', 'X8-X9 X11-X18', 'X19 X1-X4 X14-X18','X19 X1-X9', 'X1-X2 X4-X18 X20']
strengthScenario = ['No Omit', 'S', 'S', 'S','W' 'W', 'S W', 'S S','W W' '10W', '5S 5W','10S','All but 2S']


scenario = ['X999', 'X1' ]
scenario_order = ['No Omit', 'S_X1' ]

latex_table_modified = main_result_latex(df_csv, scenario, scenario_order, it_list, measures, DGP_name,  multiple_measures, label, caption)

# Write to LaTeX file
with open(f"tables/AppendixMainResult{DGP_name}_1.tex", "w") as f:
    f.write(latex_table_modified)


scenario = ['X2', 'X3', 'X17', 'X18' ]
scenario_order = ['S_X2', 'S_X3', 'W_X17', 'W_X18']
label = f"appendix{DGP_name}Result2"
caption = f"Appendix Result 2: Bias, Absolute Bias, and RMSE for {DGP_name} DGP."

latex_table_modified = main_result_latex(df_csv, scenario, scenario_order, it_list, measures, DGP_name,  multiple_measures, label, caption)

# Write to LaTeX file
with open(f"tables/AppendixMainResult{DGP_name}_2.tex", "w") as f:
    f.write(latex_table_modified)

scenario = ['X2 X18', 'X1 X2', 'X17 X18', 'X8-X9 X11-X18']
scenario_order = ['S W', 'S S', 'W W' ,'10W']
label = f"appendix{DGP_name}Result3"
caption = f"Appendix Result 3: Bias, Absolute Bias, and RMSE for {DGP_name} DGP."

latex_table_modified = main_result_latex(df_csv, scenario, scenario_order, it_list, measures, DGP_name,  multiple_measures, label, caption)

# Write to LaTeX file
with open(f"tables/AppendixMainResult{DGP_name}_3.tex", "w") as f:
    f.write(latex_table_modified)

    scenario = ['X19 X1-X4 X14-X18', 'X19 X1-X9', 'X1-X2 X4-X18 X20']
scenario_order = ['5S 5W','10S','All but 2S']
label = f"appendix{DGP_name}Result4"
caption = f"Appendix Result 4: Bias, Absolute Bias, and RMSE for {DGP_name} DGP."

latex_table_modified = main_result_latex(df_csv, scenario, scenario_order, it_list, measures, DGP_name,  multiple_measures, label, caption)

# Write to LaTeX file
with open(f"tables/AppendixMainResult{DGP_name}_4.tex", "w") as f:
    f.write(latex_table_modified)

# Function to incrementally number 'S' and 'W' only when they appear alone
def modify_scenario(group):
    # Create increasing counters for single S and W within the group
    s_counter = group['Scenario'].eq('S').cumsum()  # Count 'S'
    w_counter = group['Scenario'].eq('W').cumsum()  # Count 'W'

    # Apply the counters to single 'S' and 'W'
    new_scenarios = []
    for idx, scenario in group['Scenario'].items():
        if scenario == 'S':
            new_scenarios.append(f"S_{s_counter[idx]}")
        elif scenario == 'W':
            new_scenarios.append(f"W_{w_counter[idx]}")
        else:
            new_scenarios.append(scenario)  # Leave other cases unchanged

    return pd.Series(new_scenarios, index=group.index)  # Ensure index is maintained

# Group the DataFrame by 'x_omit' and apply the modification function
df['Scenario'] = df.groupby('x_omit', group_keys=False).apply(modify_scenario)

# Display the modified DataFrame
df['Scenario'].drop_duplicates()









# monte carlo SE of measures

def table_monte_carlo_SE(df, scenario_order, measure = 'Monte Carlo SE Bias'):
    #ml_algo_order = ['Linear Regression', 'DML', 'T-learner', 'X-learner', 'XBCF']

    # Convert the ml_algo column to a categorical type with the specified order
    #df['ml_algo'] = pd.Categorical(df['ml_algo'], categories=ml_algo_order, ordered=True)

    # rename column of df 
    df = df.rename(columns={'ml_algo': 'Method'})
    df = df.rename(columns={'Monte_carlo_SE_bias': 'Monte Carlo SE Bias'})
    df = df.rename(columns={'Monte_carlo_SE_MSE': 'Monte Carlo SE MSE'})
    df = df.rename(columns={'Monte_carlo_SE_RMSE': 'Monte Carlo SE RMSE'})
    df = df.rename(columns={'Monte_carlo_SE_abs_bias': 'Monte Carlo SE Abs Bias'})


    pivot_table = df.pivot_table(index=['DGP','Method'], columns='Scenario', values = measure)

    a = grouping_mean_df(df, ['Method','Scenario', 'DGP'], mean = True)

    total_measure = a.groupby(['DGP','Method']).sum()[[measure]]
    # Add the 'Total' column to the pivot table

    pivot_table[('Total')] = total_measure

    scenario_order.append('Total')

    # Reindex the pivot table based on the ordered columns
    pivot_table = pivot_table.reindex(columns=scenario_order)

    pivot_table = pivot_table.round(3)

    return pivot_table


scenario = ['X999', 'X1', 'X15', 'X10-X19', 'X3-X20']
scenario_order = ['No Omit', 'S', 'W', '10W', 'All but 2S'] 

df = df_csv[df_csv['x_omit'].isin(scenario)]
df = df[df['DGP'] == 'Linear']

MC_bias_linear = table_monte_carlo_SE(df, scenario_order, measure = 'Monte Carlo SE Bias')

scenario = ['X999', 'X3', 'X17', 'X8-X9 X11-X18', 'X1-X2 X4-X18 X20']
scenario_order = ['No Omit', 'S', 'W', '10W', 'All but 2S'] 

df = df_csv[df_csv['x_omit'].isin(scenario)]
df = df[df['DGP'] == 'Non-Linear']

MC_bias_non_linear = table_monte_carlo_SE(df, scenario_order, measure = 'Monte Carlo SE Bias')

# Convert pivot table to LaTeX format
latex_table_linear = MC_bias_linear.to_latex(multicolumn=True, multirow=True, label= f'tab:MonteCarloSEMain', caption=f'Result: Monte Carle SE for Bias')

latex_table_non_linear = MC_bias_non_linear.to_latex(multicolumn=True, multirow=True, label= f'tab:MonteCarloSEMain', caption=f'Result: Monte Carle SE for Bias')



buttomrule_index = latex_table_linear.find('\\bottomrule')

latex_table_first = latex_table_linear[:buttomrule_index]


midrule_index = latex_table_non_linear.find('\\midrule')
# Keep everything from the first '\midrule' onward
latex_table_last = latex_table_non_linear[midrule_index:]

# Combine the first part, custom line, and last part
latex_table_modified = latex_table_first  + latex_table_last

# Write to a file 
with open(f"tables/MonteCarloSEMain.tex", "w") as f:
    f.write(latex_table_modified)


# time calculation 
df = df_csv.rename(columns={'ml_algo': 'Method'})
pivot_table = df.pivot_table(index=['Method'], columns='DGP', values='time')


pivot_table = pivot_table.round(3)


# Write to a file
latex_table = pivot_table.to_latex(label='tab:time', caption=f'Execution time for Methods.')
with open(f"tables/time.tex", "w") as f:
    f.write(latex_table)



# ---------------------------------------------------------------------------------------------------
# DML special results

# large observations

df = df_DML

df = df.rename(columns={'ml_algo': 'Method', 'abs_bias': 'Abs Bias', 'n_observations': 'N Obs', 'bias':'Bias'})

df['ScenarioX'] = df.apply(lambda row: row['Scenario'] + '_' + row['x_omit'] 
                    if row['Scenario'] in ['S', 'W'] else row['Scenario'], axis=1)

df = df[['ScenarioX', 'Bias', 'Abs Bias', 'RMSE']].round(4)

latex_table = df.to_latex(index=False, label='tab:DMLResultsManyObs', caption='Appendix: DML Results for 30000 Observations.')


# Add the small text right after the table and before \end{table}
custom_text = "\\\\\small{S and W correspond to Strong and Weak. Prop. Function stands for the underlying ML method used to estimate the nuisance function of propensity score. The DGP specifications are: Non-Linear DGP, lambda = 0.5, gamma = 0.5.}"

# Combine the LaTeX code for the table and the custom text
latex_table_with_text = latex_table.replace("\\end{table}", custom_text + "\n\\end{table}")

# Write the combined LaTeX to a file
with open(f'tables/DMLResultsManyObs.tex', "w") as f:
    f.write(latex_table_with_text)

'

# DML different propensity score
df = df_DML_prop

df = df.rename(columns={'ml_algo': 'Method', 'abs_bias': 'Abs Bias', 'n_observations': 'N Obs', 'bias':'Bias', 'propensity_nuisance' : 'Prop. Function'})

df['ScenarioX'] = df.apply(lambda row: row['Scenario'] + '_' + row['x_omit'] 
                    if row['Scenario'] in ['S', 'W'] else row['Scenario'], axis=1)

order_index = ['No Omit', 'S_X1', 'S_X2', 'S_X3', 'W_X17', 'W_X18', 'S W', 'S S','W W', '10W', '5S 5W','10S','All but 2S']
df['ScenarioX'] = pd.Categorical(df['ScenarioX'], categories=order_index, ordered=True)

pivot_table = df.pivot_table(
    index=['ScenarioX', 'Prop. Function'],  
    values=['Bias', 'Abs Bias', 'RMSE'],    
    aggfunc='mean'                          
)

latex_table = pivot_table.to_latex(index=True, label='tab:DMLResultsPropensity', caption='Appendix: DML Results for different Propensity.')


# Add the small text right after the table and before \end{table}
custom_text = "\\\\\small{S and W correspond to Strong and Weak. Prop. Function stands for the underlying ML method used to estimate the nuisance function of propensity score. The DGP specifications are: Non-Linear DGP, lambda = 0.5, gamma = 0.5, N Obs = 8000.}"

# Combine the LaTeX code for the table and the custom text
latex_table_with_text = latex_table.replace("\\end{table}", custom_text + "\n\\end{table}")

# Write the combined LaTeX to a file
with open(f'tables/DMLResultsPropensity.tex', "w") as f:
    f.write(latex_table_with_text)



# artificial DGP simulation

# duplicate test
# Assuming df is your DataFrame

df_model_filter = df_csv[(df_csv['linear_di'] == False) & (df_csv['linear_pi'] == False)]

df_model_filter = df_csv[(df_csv['linear_di'] == True) & (df_csv['linear_pi'] == False)]

# Assuming df is your DataFrame
unique_combinations = df_model_filter[['n_observations', 'n_conf', 'gamma']].drop_duplicates()

# If you want it as a list of tuples
unique_combinations_list = [tuple(x) for x in unique_combinations.to_numpy()]

len(unique_combinations_list)

indicator_linear_di = df_model_filter['linear_di'].unique()[0]

# Use the 'science' style
plt.style.use(['science', 'no-latex'])


def filter_function(df, n_conf, n_observation, gamma):
    return df[(df['n_conf'] == n_conf) & (df['n_observations'] == n_observation) & (df['gamma'] == gamma)]


def plot_change_y_axis(df_filtered, ax, y_string, SE_string, i, n_conf, n_observation, gamma, text_position: str, SE_plot = True, add_text = True):
    # Plot the line plot
    sns_plot = sns.lineplot(ax=ax, data=df_filtered, x='share_omit', y=y_string, hue='ml_algo', marker=' ', 
                            err_style='bars', errorbar=None, legend='brief')  # Enable a brief legend

    if SE_plot:
        # Overlay the scatter plot with varying dot sizes
        for name, group in df_filtered.groupby('ml_algo'):
            # Get the color of the current line
            line_color = sns.color_palette()[list(df_filtered['ml_algo'].unique()).index(name)]

            def adjust_size_SE(gamma):
                if gamma <= float(0.2):
                    return 9000
                elif gamma <= float(0.5):
                    return 6000
                elif gamma <= float(2):
                    return 2000
            if y_string == 'bias':
                adjuster = 1 * (1/gamma)
            else:
                adjuster = 0.2 * (1/gamma)
            size_SE = adjust_size_SE(gamma)
            # Plot the scatter plot with the same color on the correct axis
            ax.scatter(group['share_omit'], group[y_string], 
                    s=group[SE_string] * int(size_SE) * adjuster,  # Adjust the multiplier for larger dots
                    color=line_color, alpha=0.7)

    # Collect legend handles and labels from the first subplot
    if i == 0:
        handles, labels = ax.get_legend_handles_labels()
    else:
        handles, labels = [], []

    # Add titles and labels
    ax.set_xlabel('')  # Remove x-axis label
    ax.set_ylabel(y_string, fontsize=10)  # Set y-axis label with font size 8

    # Reduce font size of x-axis and y-axis tick labels
    ax.tick_params(axis='both', which='major', labelsize=9)  # Reduce tick size for both axes

    # Remove the legend from individual plots
    ax.legend([], [], frameon=False)

    if add_text:
        if text_position == 'upper':
            # Add text to the upper right side of the plot
            ax.text(0.73, 0.92, f'Additional Info:\n- N conf: {n_conf}\n- N obs: {n_observation}\n- gamma: {gamma}',
                    fontsize=9, ha='left', va='top', transform=ax.transAxes)
        else:
            # Add text to the lower right side of the plot
            ax.text(0.73, 0.05, f'Additional Info:\n- N conf: {n_conf}\n- N obs: {n_observation}\n- gamma: {gamma}',
                    fontsize=9, ha='left', va='bottom', transform=ax.transAxes)
    
    return ax, handles, labels


sim_counter = 0
number_of_pages = int(len(unique_combinations_list) / 3)

#number_of_pages = 1  # Just for testing
for i_page in range(number_of_pages):


    # Set up the A4 figure with subplots
    fig, axes = plt.subplots(3, 2, figsize=(9.7, 11.69))  # exact A4 size in inches

    # Flatten the axes array for easy iteration
    axes = axes.flatten()

    # Initialize an empty list to collect handles and labels
    handles, labels = [], []
    # Loop over each subplot and plot the data
    for i, ax in enumerate(axes):
        n_observation = int(unique_combinations_list[sim_counter][0])
        n_conf = int(unique_combinations_list[sim_counter][1])
        gamma = unique_combinations_list[sim_counter][2]

        df_filtered = filter_function(df_model_filter, n_conf, n_observation, gamma)

        if i % 2 == 0:
            # placing first plot
            if df_model_filter['linear_di'].unique()[0] == True:
                    text_position = 'upper'
            else:
                text_position = None
            ax, h, l = plot_change_y_axis(df_filtered, ax, 'bias', 'Monte_carlo_SE_bias', i, n_conf, n_observation, gamma, text_position=text_position)
        else:
            # placing second plot
            ax, h, l = plot_change_y_axis(df_filtered, ax, 'MSE', 'Monte_carlo_SE_MSE', i, n_conf, n_observation, gamma, text_position=None)
            # go to the next simulation after placing the MSE plot
            sim_counter += 1

        if i == 0:
            handles, labels = h, l

    # Place a combined legend above the entire plot
    legend = fig.legend(
        handles=handles,
        labels=labels,
        title='Methods', # \nX-axis equals the share of omitted variables. \nSize of the round circles along the lines represent the Monte Carlo standard errors of the measure at display',
        loc='upper center',
        bbox_to_anchor=(0.5, 1.01),
        fancybox=True,
        shadow=True,
        ncol=5,
        prop={'size': 9}
    )

    # Center the title and reduce its font size
    legend.get_title().set_fontsize(10)  # Set smaller font size
    legend.get_title().set_ha('center')  # Center the title horizontally

    # Adjust layout to fit plots within the figure area
    plt.tight_layout(rect=[0, 0, 1, 0.98])

    # Save the figure as a PDF
    fig.savefig(f'visuals/linear_di{indicator_linear_di}_page{i_page}.pdf', format='pdf', bbox_inches='tight')

# Show the plot (optional if you're saving as PDF)
#plt.show()
plt.close()








# real world simulation

df_ihdp = pd.read_csv(source_path + '/Data-IHDP/ihdp.csv')
n_observation = len(df_ihdp['bw'])
n_conf = len(df_ihdp.columns) # without the treatment variable


# Assuming df is your DataFrame
unique_combinations = df_real_world[['n_observations', 'n_conf', 'gamma']].drop_duplicates()

# If you want it as a list of tuples
unique_combinations_list = [tuple(x) for x in unique_combinations.to_numpy()]

def plot_change_y_axis(df_filtered, ax, y_string, SE_string, i, n_conf, n_observation, gamma, text_position: str):
    # Plot the line plot
    sns_plot = sns.lineplot(ax=ax, data=df_filtered, x='share_omit', y=y_string, hue='ml_algo', marker=' ', 
                            err_style='bars', errorbar=None, legend='brief')  # Enable a brief legend

    # Overlay the scatter plot with varying dot sizes
    for name, group in df_filtered.groupby('ml_algo'):
        # Get the color of the current line
        line_color = sns.color_palette()[list(df_filtered['ml_algo'].unique()).index(name)]

        def adjust_size_SE(gamma):
            if gamma <= float(0.1):
                return 7000
            elif gamma <= float(0.5):
                return 4000
        if y_string == 'bias':
            adjuster = 0.8 * (1/gamma)
        else:
            adjuster = 0.5 * (1/gamma)
        size_SE = adjust_size_SE(gamma)
        # Plot the scatter plot with the same color on the correct axis
        ax.scatter(group['share_omit'], group[y_string], 
                   s=group[SE_string] * int(size_SE) * adjuster,  # Adjust the multiplier for larger dots
                   color=line_color, alpha=0.7)

    # Collect legend handles and labels from the first subplot
    if i == 0:
        handles, labels = ax.get_legend_handles_labels()
    else:
        handles, labels = [], []

    # Add titles and labels
    ax.set_xlabel('')  # Remove x-axis label
    ax.set_ylabel(y_string, fontsize=10)  # Set y-axis label with font size 8

    # Reduce font size of x-axis and y-axis tick labels
    ax.tick_params(axis='both', which='major', labelsize=9)  # Reduce tick size for both axes

    # Remove the legend from individual plots
    ax.legend([], [], frameon=False)

    if text_position == 'upper':
        # Add text to the upper right side of the plot
        ax.text(0.73, 0.92, f'Additional Info:\n- N conf: {n_conf}\n- N obs: {n_observation}\n- gamma: {gamma}',
                fontsize=9, ha='left', va='top', transform=ax.transAxes)
    else:
        # Add text to the lower right side of the plot
        ax.text(0.73, 0.05, f'Additional Info:\n- N conf: {n_conf}\n- N obs: {n_observation}\n- gamma: {gamma}',
                fontsize=9, ha='left', va='bottom', transform=ax.transAxes)
    
    return ax, handles, labels


sim_counter = 0
number_of_pages = int(len(unique_combinations_list) / 3)

#number_of_pages = 1  # Just for testing
for i_page in range(number_of_pages):


    # Set up the A4 figure with subplots
    fig, axes = plt.subplots(3, 2, figsize=(9.7, 11.69))  # exact A4 size in inches

    # Flatten the axes array for easy iteration
    axes = axes.flatten()

    # Initialize an empty list to collect handles and labels
    handles, labels = [], []
    # Loop over each subplot and plot the data
    for i, ax in enumerate(axes):
        gamma = unique_combinations_list[sim_counter][2]

        df_filtered = filter_function(df_real_world, n_conf = 5, n_observation = 100,  gamma = gamma)

        if i % 2 == 0:
            # placing first plot
            ax, h, l = plot_change_y_axis(df_filtered, ax, 'bias', 'Monte_carlo_SE_bias', i, n_conf, n_observation, gamma, text_position=None)
        else:
            # placing second plot
            ax, h, l = plot_change_y_axis(df_filtered, ax, 'MSE', 'Monte_carlo_SE_MSE', i, n_conf, n_observation, gamma, text_position=None)
            # go to the next simulation after placing the MSE plot
            sim_counter += 1

        if i == 0:
            handles, labels = h, l

    # Place a combined legend above the entire plot
    legend = fig.legend(
        handles=handles,
        labels=labels,
        title='Methods', # \nX-axis equals the share of omitted variables. \nSize of the round circles along the lines represent the Monte Carlo standard errors of the measure at display',
        loc='upper center',
        bbox_to_anchor=(0.5, 1.01),
        fancybox=True,
        shadow=True,
        ncol=5,
        prop={'size': 9}
    )

    # Center the title and reduce its font size
    legend.get_title().set_fontsize(10)  # Set smaller font size
    legend.get_title().set_ha('center')  # Center the title horizontally

    # Adjust layout to fit plots within the figure area
    plt.tight_layout(rect=[0, 0, 1, 0.98])

    # Save the figure as a PDF
    fig.savefig(f'visuals/real_world_page{i_page}.pdf', format='pdf', bbox_inches='tight')

# Show the plot (optional if you're saving as PDF)
#plt.show()
plt.close()




##############################################################################
# visualization for aggregated results

import matplotlib.pyplot as plt
import seaborn as sns

# Simplified version of the function
def plot_change_y_axis(df_filtered, ax, y_string, SE_string, i):
    # Plot the line plot
    sns_plot = sns.lineplot(ax=ax, data=df_filtered, x='share_omit', y=y_string, hue='ml_algo', marker=' ', 
                            err_style='bars', errorbar=None, legend='brief')  # Enable a brief legend

    # Collect legend handles and labels from the first subplot
    if i == 0:
        handles, labels = ax.get_legend_handles_labels()
    else:
        handles, labels = [], []

    # Add titles and labels
    ax.set_xlabel('')  # Remove x-axis label
    ax.set_ylabel(y_string, fontsize=10)  # Set y-axis label with font size 8

    # Reduce font size of x-axis and y-axis tick labels
    ax.tick_params(axis='both', which='major', labelsize=9)  # Reduce tick size for both axes

    # Remove the legend from individual plots
    ax.legend([], [], frameon=False)

    return ax, handles, labels

def single_create_plots(df_real_world, sim_counter=0, figsize=(8, 5), legend=True, name='real_world'):
   
    # Set up the A4 figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=figsize)  # Set up figure with subplots

    # Flatten the axes array for easy iteration
    axes = axes.flatten()

    # Initialize an empty list to collect handles and labels
    handles, labels = [], []

    # Loop over each subplot and plot the data
    for i, ax in enumerate(axes):
        if i % 2 == 0:
            # Plot the first type (bias)
            ax, h, l = plot_change_y_axis(df_real_world, ax, 'bias', 'Monte_carlo_SE_bias', i)
        else:
            # Plot the second type (MSE)
            ax, h, l = plot_change_y_axis(df_real_world, ax, 'MSE', 'Monte_carlo_SE_MSE', i)
            # Increment the simulation counter after placing the MSE plot
            sim_counter += 1

        if i == 0:
            handles, labels = h, l  # Capture the legend handles and labels from the first plot

    # Place a combined legend above the entire plot
    if legend:
        legend = fig.legend(
            handles=handles,
            labels=labels,
            title='Methods',  # You can customize the legend title here
            loc='upper center',
            bbox_to_anchor=(0.5, 1.04),  # Move the legend above the plot
            fancybox=True,
            shadow=True,
            ncol=5,
            prop={'size': 9}
        )

        # Center the legend title and adjust font size
        legend.get_title().set_fontsize(10)
        legend.get_title().set_ha('center')

    # Adjust layout to make room for the legend above the plot
    plt.subplots_adjust(top=0.85)  # Adjust the top to make space for the legend

    # Save the figure as a PDF
    fig.savefig(f'visuals/main_visual_{name}.pdf', format='pdf', bbox_inches='tight')

    # Optionally show the plot
    plt.show()

plt.close()


df_all_mean = union_df.groupby(['ml_algo', 'share_omit']).mean().reset_index()

df_semi_linear = df_csv[df_csv['linear_di'] == True]
df_non_linear = df_csv[df_csv['linear_di'] == False]

def grouping_mean_df(df, group_by, mean = True):
    if mean == True:
        return df.groupby(group_by).mean().reset_index()
    else:
        return df.groupby(group_by).sum().reset_index()


# Example usage with the test function
single_create_plots(df_non_linear, sim_counter=0, figsize=(7, 3), legend=True, name = 'non_linear')
single_create_plots(df_semi_linear, sim_counter=0, figsize=(7, 3), legend=True, name = 'semi_linear')
single_create_plots(df_real_world, sim_counter=0, figsize=(7, 3), legend=True, name = 'real_world')

grouped_real_world = grouping_mean_df(df_real_world, ['ml_algo', 'share_omit'])  
single_create_plots(grouped_real_world, sim_counter=0, figsize=(7, 3), legend=True)


# show monte carlo aggregated SE for all simulations
df = union_df


def table_monte_carlo_SE(df, measure = 'Monte Carlo SE Bias'):
    #ml_algo_order = ['Linear Regression', 'DML', 'T-learner', 'X-learner', 'XBCF']

    # Convert the ml_algo column to a categorical type with the specified order
    #df['ml_algo'] = pd.Categorical(df['ml_algo'], categories=ml_algo_order, ordered=True)

    # rename column of df 
    df = df.rename(columns={'ml_algo': 'Method'})
    df = df.rename(columns={'share_omit': 'Share Omit'})
    df = df.rename(columns={'Monte_carlo_SE_bias': 'Monte Carlo SE Bias'})
    df = df.rename(columns={'Monte_carlo_SE_MSE': 'Monte Carlo SE MSE'})


    pivot_table = df.pivot_table(index=['DGP','Method'], columns='Share Omit', values = measure)

    a = grouping_mean_df(df, ['Method','Share Omit', 'DGP'], mean = True)

    total_measure = a.groupby(['DGP','Method']).sum()[[measure]]
    # Add the 'Total' column to the pivot table

    pivot_table[('Total')] = total_measure

    pivot_table = pivot_table.round(3)
    return pivot_table

df = union_df[union_df['share_omit'] <= 0.8]

MC_bias = table_monte_carlo_SE(df, measure = 'Monte Carlo SE Bias')
MC_MSE = table_monte_carlo_SE(df, measure = 'Monte Carlo SE MSE')

list_MC = [MC_bias, MC_MSE]
namelist = ['MC_bias', 'MC_MSE']

for i, name in zip(list_MC, namelist):
# Write to a file
    latex_table = i.to_latex(label=f'tab:{name}', caption=f'Monte Carlo standard errors for {name[3:]}.')    
    with open(f"tables/{name}.tex", "w") as f:
        f.write(latex_table)


# time calculation 
df = df.rename(columns={'ml_algo': 'Method'})
pivot_table = df.pivot_table(index=['Method'], columns='DGP', values='time')


pivot_table = pivot_table.round(3)


# Write to a file
latex_table = pivot_table.to_latex(label='tab:time', caption=f'Time calculation for the different methods.')
with open(f"tables/time.tex", "w") as f:
    f.write(latex_table)





# create df with overview of the mathods

# Define the columns and index
columns = ['Method', 'description', 'package', 'underlying model']
index = ['Linear Regression', 'DML', 'T-learner', 'X-learner', 'XBCF']

# Create the data for each machine learning method
data = {
    'ML-Method': ['Linear Regression', 'DML', 'T-learner', 'X-learner', 'XBCF'],
    'description': [
        'A basic linear model to estimate treatment effects.',
        'Double machine learning using cross-fitting and two nuisance functions that are combined in a final model.',
        'Estimates nuisance function for outcome in treatment and control group.',
        'Multiple step approach with etsimation of the imputed treatment effect.',
        'Accelerated Bayesian causal forest method which is faster than the original BCF.'
    ],
    'package': [
        'statsmodels', 'doubleml', 'causalml', 'causalml', 'xbcausalforest'
    ],
    'underlying model': [
        'Ordinary Least Squares (OLS)',
        'nuisance function for outcome: XGBoost, nuisance function for propensity score: XGBoost',
        'nuisance function for outcome: XGBoost',
        'nuisance function for outcome: XGBoost, nuisance function for propensity score: Elastic Net',
        'Bayesian regression trees'
    ]
}

# Create the DataFrame
df = pd.DataFrame(data, index=index)

# Convert to LaTeX with to_latex (using the tabular environment)
latex_table = df.to_latex(index=False, escape=False, column_format='|l|p{4cm}|p{2cm}|p{4cm}|')

# Manually adjust the LaTeX table and wrap it in a correct tabularx environment
latex_table = '''
\\begin{table}[htbp]
\\centering
\\renewcommand{\\arraystretch}{1.5}
\\begin{tabularx}{\\textwidth}{|l|p{4cm}|p{2cm}|p{4cm}|}
\\hline
''' + latex_table[latex_table.find('\\hline'):] + '''
\\end{tabularx}
\\caption{Comparison of machine learning methods.}
\\end{table}
'''

# Save to file
with open("tables/ml_methods_table.tex", "w") as f:
    f.write(latex_table)

