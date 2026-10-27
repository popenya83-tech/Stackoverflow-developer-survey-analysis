import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

sns.set_theme(style="whitegrid")

pdf_report = PdfPages('stack_overflow_analysis_report.pdf')

df = pd.read_csv('survey_results_public.csv')
schema_df = pd.read_csv('survey_results_schema.csv')

df.columns = df.columns.str.strip()

qname = schema_df['qname']
questions = list(set(qname.unique()) & set(df.columns))
df_complete = df.dropna(subset=questions)

total_respondents = df['ResponseId'].nunique()
complete_respondents = df_complete['ResponseId'].nunique()
incomplete_respondents = total_respondents - complete_respondents

print(f"Total respondents: {total_respondents}")
print(f"Complete respondents: {complete_respondents}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
bars = ax1.bar(['Total', 'Complete'], [total_respondents, complete_respondents], color=['#3498db', '#2ecc71'])
ax1.set_title('Respondent Count', fontsize=14, fontweight='bold')
ax1.set_ylabel('Number of People')

for bar in bars:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + (total_respondents * 0.01),
             f'{int(height)}', ha='center', va='bottom', fontsize=12)

ax2.pie([complete_respondents, incomplete_respondents],
        labels=['Complete', 'Incomplete'],
        autopct='%1.1f%%',
        startangle=140,
        colors=['#2ecc71', '#e74c3c'],
        explode=(0.1, 0))
ax2.set_title('Survey Completion Rate', fontsize=14, fontweight='bold')
plt.tight_layout()
pdf_report.savefig()
plt.close()

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(['Initial Count', 'After Cleaning'],
              [total_respondents, complete_respondents],
              color=['#bdc3c7', '#2ecc71'])

for bar in bars:
    height = bar.get_height()
    ax.annotate(f'{height}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=12, fontweight='bold')

info_text = (
    f"Analyzed Questions: {len(questions)}\n"
    f"Dropped (due to NaN): {total_respondents - complete_respondents}\n"
    f"Retained: {((complete_respondents/total_respondents)*100):.1f}% of data"
)

plt.text(1.5, total_respondents * 0.8, info_text, fontsize=11,
         bbox=dict(facecolor='white', alpha=0.5, edgecolor='black', boxstyle='round,pad=1'))

plt.title('Sample Cleaning Results (Listwise Deletion)', fontsize=16, pad=20)
plt.ylabel('Number of Respondents')
plt.ylim(0, total_respondents * 1.15)
plt.tight_layout()
pdf_report.savefig()
plt.close()

mean_val = df['WorkExp'].mean()
median_val = df['WorkExp'].median()
mode_val = df['WorkExp'].mode()[0]

print(f"\nCentral Tendency Measures for WorkExp:")
print(f"---")
print(f"Mean:   {round(mean_val, 1)}")
print(f"Median: {median_val}")
print(f"Mode:   {mode_val}")

plt.figure(figsize=(12, 6))
sns.histplot(df['WorkExp'].dropna(), kde=True, color='skyblue', bins=30)
plt.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.1f}')
plt.axvline(median_val, color='green', linestyle='-', linewidth=2, label=f'Median: {median_val}')
plt.axvline(mode_val, color='orange', linestyle='-.', linewidth=2, label=f'Mode: {mode_val}')
plt.title('Work Experience Distribution (WorkExp)', fontsize=15)
plt.xlabel('Years of Experience', fontsize=12)
plt.ylabel('Number of Respondents', fontsize=12)
plt.legend()
plt.tight_layout()
pdf_report.savefig()
plt.close()

remote_workers_count = df[df['RemoteWork'] == 'Remote'].shape[0]
print(f"\nNumber of remote workers: {remote_workers_count}")

work_format_counts = df['RemoteWork'].value_counts()
plt.figure(figsize=(12, 6))
plot = sns.barplot(
    y=work_format_counts.index,
    x=work_format_counts.values,
    hue=work_format_counts.index,
    palette="viridis",
    legend=False
)
plt.title('Work Format Distribution', fontsize=16, pad=20)
plt.xlabel('Number of Respondents', fontsize=12)
plt.ylabel('Work Format', fontsize=12)

for i, v in enumerate(work_format_counts.values):
    plot.text(v + (max(work_format_counts.values) * 0.01), i, f'{int(v)}',
              color='black', va='center', fontweight='bold')

plt.tight_layout()
pdf_report.savefig()
plt.close()

df['worked_with_python'] = df['LanguageHaveWorkedWith'].str.contains('Python', case=False, na=False)
python_users_count = df['worked_with_python'].sum()
python_percentage = (python_users_count / len(df)) * 100
print(f"Percentage of Python users: {python_percentage:.1f}%")

python_stats = df['worked_with_python'].value_counts()
labels = ['Other Languages', 'Use Python']
sizes = [python_stats[False], python_stats[True]]
colors = ['#ced4da', '#3776ab']
explode = (0, 0.1)

plt.figure(figsize=(8, 8))
plt.pie(
    sizes,
    explode=explode,
    labels=labels,
    autopct='%1.1f%%',
    shadow=True,
    startangle=140,
    colors=colors,
    textprops={'fontsize': 14, 'fontweight': 'bold'}
)
plt.title('Share of Respondents Using Python', fontsize=16, pad=20)
plt.tight_layout()
pdf_report.savefig()
plt.close()

df['learned_with_online_courses'] = df['LearnCode'].str.contains('Online Courses', na=False)
online_learners_count = df['learned_with_online_courses'].sum()
print(f"Number of respondents who learned via online courses: {online_learners_count}")

counts = df['learned_with_online_courses'].value_counts()
plt.figure(figsize=(8, 6))
ax = sns.barplot(x=counts.index, y=counts.values, hue=counts.index, palette='viridis', legend=False)
plt.title('Distribution of Respondents Using Online Courses', fontsize=15)
plt.xlabel('Used Online Courses', fontsize=12)
plt.ylabel('Number of Respondents', fontsize=12)
plt.xticks([0, 1], ['No (False)', 'Yes (True)'])

for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}',
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center',
                xytext=(0, 9),
                textcoords='offset points')

plt.tight_layout()
pdf_report.savefig()
plt.close()

py_devs = df[df['LanguageHaveWorkedWith'].str.contains('Python', na=False)].copy()
py_devs = py_devs.dropna(subset=['ConvertedCompYearly'])

py_progers_by_country = py_devs.groupby('Country')['ConvertedCompYearly'].agg(['mean', 'median'])
py_progers_by_country = py_progers_by_country.sort_values(by='median', ascending=False)
py_progers_by_country.columns = ['Mean Compensation', 'Median Compensation']

print("\nTop 5 countries by Python developer compensation (Median):")
print(py_progers_by_country.head(5))

top_10_countries = py_progers_by_country.head(10).copy()
plt.figure(figsize=(12, 8))
top_10_countries.plot(kind='barh', width=0.8, ax=plt.gca())
plt.title('Top 10 Countries by Python Developer Compensation', fontsize=16, pad=20)
plt.xlabel('Yearly Compensation (USD)', fontsize=12)
plt.ylabel('Country', fontsize=12)
plt.legend(['Mean', 'Median'], loc='lower right')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.gca().invert_yaxis()
plt.tight_layout()
pdf_report.savefig()
plt.close()

top_paid_df = df[['ResponseId', 'EdLevel', 'ConvertedCompYearly']].dropna(subset=['ConvertedCompYearly'])
top_paid_df = top_paid_df.sort_values(by='ConvertedCompYearly', ascending=False).reset_index(drop=True)
top_5_education = top_paid_df.head(5)

print("\nEducation levels of top 5 highest-paid respondents:")
print(top_5_education[['ResponseId', 'EdLevel', 'ConvertedCompYearly']])

plt.figure(figsize=(12, 7))
ax = sns.barplot(
    x='ConvertedCompYearly',
    y='EdLevel',
    data=top_5_education,
    hue='EdLevel',
    palette='viridis',
    legend=False
)

for p in ax.patches:
    width = p.get_width()
    plt.text(
        width + 5000,
        p.get_y() + p.get_height()/2,
        f'${width:,.0f}',
        va='center',
        fontsize=11,
        fontweight='bold'
    )

plt.title('Top 5 Education Levels by Yearly Compensation', fontsize=16, pad=20, fontweight='bold')
plt.xlabel('Yearly Compensation (USD)', fontsize=12)
plt.ylabel('Education Level', fontsize=12)
plt.xlim(0, top_5_education['ConvertedCompYearly'].max() * 1.2)
sns.despine(left=True, bottom=True)
plt.tight_layout()
pdf_report.savefig()
plt.close()

df_filtered = df.dropna(subset=['Age', 'LanguageHaveWorkedWith']).copy()
df_filtered['IsPythonUser'] = df_filtered['LanguageHaveWorkedWith'].str.contains('Python', case=True, na=False)

age_grouping = df_filtered.groupby('Age')['IsPythonUser'].agg(['count', 'sum'])
age_grouping['Percentage'] = (age_grouping['sum'] / age_grouping['count']) * 100
python_popularity_table = age_grouping[['Percentage']].reset_index()
python_popularity_table.columns = ['Age Group', 'Python User Percentage']
python_popularity_table['Python User Percentage'] = python_popularity_table['Python User Percentage'].round(2)

print("\nPython popularity by age group:")
print(python_popularity_table)

plt.figure(figsize=(12, 7))
plot_df = python_popularity_table.sort_values('Python User Percentage', ascending=False)

ax = sns.barplot(
    x='Python User Percentage',
    y='Age Group',
    data=plot_df,
    hue='Age Group',
    palette='viridis',
    legend=False
)

for p in ax.patches:
    width = p.get_width()
    plt.text(
        width + 1.5,
        p.get_y() + p.get_height()/2,
        f'{width:.1f}%',
        va='center',
        fontsize=12,
        fontweight='bold'
    )

plt.title('Python Popularity Across Age Groups', fontsize=18, pad=25, fontweight='bold')
plt.xlabel('Percentage of Respondents (%)', fontsize=14)
plt.ylabel('Age Group', fontsize=14)
plt.xlim(0, 100)
sns.despine(left=True, bottom=True)
plt.tight_layout()
pdf_report.savefig()
plt.close()

q75_comp = df['ConvertedCompYearly'].quantile(0.75)
high_earning_remote = df[
    (df['ConvertedCompYearly'] > q75_comp) &
    (df['RemoteWork'] == 'Remote')
].copy()

industry_stats = high_earning_remote['Industry'].dropna().value_counts().reset_index()
industry_stats.columns = ['Industry', 'Count']

print(f"\n75th percentile compensation threshold: ${q75_comp:,.2f}")
print("Top industries among high-earning remote professionals:")
print(industry_stats.head(5))

plt.figure(figsize=(12, 8))
ax = sns.barplot(
    x='Count',
    y='Industry',
    data=industry_stats,
    hue='Industry',
    palette='rocket',
    legend=False
)

for p in ax.patches:
    width = p.get_width()
    plt.text(
        width + (industry_stats['Count'].max() * 0.01),
        p.get_y() + p.get_height()/2,
        f'{int(width)}',
        va='center',
        fontsize=11,
        fontweight='bold'
    )

plt.title('Most Popular Industries Among High-Earning Remote Specialists\n(Top 25% by Income)',
          fontsize=16, pad=25, fontweight='bold')
plt.xlabel('Number of Respondents', fontsize=12, labelpad=15)
plt.ylabel('Industry', fontsize=12, labelpad=15)
plt.xlim(0, industry_stats['Count'].max() * 1.15)
sns.despine(left=True, bottom=True)
plt.tight_layout()
pdf_report.savefig()
plt.close()

pdf_report.close()
print("\nSuccessfully created single PDF report: stack_overflow_analysis_report.pdf")
