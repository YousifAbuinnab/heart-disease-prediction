# Step-by-Step Guide: Pushing to GitHub

## Prerequisites
- GitHub account (create one at https://github.com if you don't have one)
- Git installed on your computer (check with `git --version`)

---

## Step 1: Create a GitHub Repository

1. Go to https://github.com and sign in
2. Click the **"+"** icon in the top right corner
3. Select **"New repository"**
4. Fill in the details:
   - **Repository name**: `heart-disease-prediction` (or any name you prefer)
   - **Description**: "Heart Disease Prediction using Machine Learning - Logistic Regression, SVM, Neural Network"
   - **Visibility**: Choose Public or Private
   - **DO NOT** check "Initialize this repository with a README" (we already have one)
   - **DO NOT** add .gitignore or license (we already have them)
5. Click **"Create repository"**
6. **Copy the repository URL** that GitHub shows you (it will look like: `https://github.com/yourusername/heart-disease-prediction.git`)

---

## Step 2: Initialize Git in Your Project

Open PowerShell/Terminal in your project folder and run:

```bash
git init
```

This creates a new Git repository in your project folder.

---

## Step 3: Add All Files to Git

```bash
git add .
```

This stages all your files (README.md, Python script, requirements.txt, visualizations, etc.)

---

## Step 4: Create Your First Commit

```bash
git commit -m "Initial commit: Heart Disease Prediction ML Project"
```

This saves all your files as the first version of your project.

---

## Step 5: Connect to GitHub Repository

Replace `YOUR_REPO_URL` with the URL you copied from Step 1:

```bash
git remote add origin YOUR_REPO_URL
```

Example:
```bash
git remote add origin https://github.com/yourusername/heart-disease-prediction.git
```

---

## Step 6: Rename Branch to Main (if needed)

```bash
git branch -M main
```

This ensures your branch is named "main" (GitHub's default).

---

## Step 7: Push to GitHub

```bash
git push -u origin main
```

You'll be prompted to enter your GitHub username and password (or personal access token).

**Note**: If you use two-factor authentication, you'll need to create a Personal Access Token instead of using your password.

---

## Troubleshooting

### If you get authentication errors:
1. Go to GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)
2. Generate a new token with `repo` permissions
3. Use this token as your password when pushing

### If you need to update your repository later:
```bash
git add .
git commit -m "Your commit message"
git push
```

---

## Quick Command Summary

```bash
# 1. Initialize
git init

# 2. Add files
git add .

# 3. Commit
git commit -m "Initial commit: Heart Disease Prediction ML Project"

# 4. Add remote (replace with your URL)
git remote add origin https://github.com/yourusername/heart-disease-prediction.git

# 5. Rename branch
git branch -M main

# 6. Push
git push -u origin main
```

