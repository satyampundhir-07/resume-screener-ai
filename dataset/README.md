# Dataset Sources

## Recommended Kaggle Datasets

### 1. Resume Dataset
**URL:** https://www.kaggle.com/datasets/gauravduttakiit/resume-dataset
- 2,400+ resumes across 24 job categories
- Fields: `Resume`, `Category`
- Format: CSV

### 2. Job Description Dataset
**URL:** https://www.kaggle.com/datasets/ravindrasinghrana/job-description-dataset
- 5,000+ real job postings
- Fields: `Job Title`, `Job Description`, `skills`, `Company`
- Format: CSV

### 3. UpdatedResumeDataSet
**URL:** https://www.kaggle.com/datasets/avishekmajumder/resumedataset
- Cleaned resume text with category labels
- Format: CSV

## How to Download (Kaggle CLI)
```bash
pip install kaggle
# Place your kaggle.json API key in ~/.kaggle/
kaggle datasets download -d gauravduttakiit/resume-dataset -p dataset/
kaggle datasets download -d ravindrasinghrana/job-description-dataset -p dataset/
```

## sample_data.csv Format
The project includes `sample_data.csv` with 5 hand-crafted resume-JD pairs
for demonstration. Each row has:

| Column | Description |
|--------|-------------|
| `resume` | Plain-text resume content |
| `jd` | Plain-text job description |
| `label` | Similarity label 0.0–1.0 (ground truth for evaluation) |
