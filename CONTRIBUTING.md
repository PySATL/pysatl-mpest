# Pysatl-mpest project contributing guide

Thank you very much if you have decided to contribute to our project.

## How to Contribute to pysatl-mpest

Follow these steps to get started. We use the Forking Workflow model.

### 1. Fork the Repository

First, you need to fork (create a copy of) the main repository.

1. Navigate to the repository page: https://github.com/PySATL/pysatl-mpest
2. Click the "Fork" button in the top-right corner of the page. This will create a copy of the repository in your GitHub account.



### 2. Clone the Fork and Create a Branch

Now that you have a fork, clone it to your local machine:

```bash
git clone https://github.com/YOUR-USERNAME/pysatl-mpest.git
cd pysatl-mpest
```

Before you start making changes, create a new branch in following format `<type>/<what-this-branch-solves>`.

For example: `feat/add-new-solver` or `fix/memory-leak-issue`.

```bash
git checkout main
git checkout -b <type>/<what-this-branch-solves>
```



### 3. Make and Commit Changes

Work on the code in your new branch. When you're ready to save your changes, create a commit. We use [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) to standardize commit messages.

Use the following types:

- `feat`: to add new functionality.
- `fix`: to fix a bug in the project.
- `refactor`: for code refactoring, such as renaming a variable or improving readability.
- `test`: to add or refactor tests.
- `struct`: for changes related to the project structure (NOT CODE), for example, changing folder locations.
- `ci`: for various CI/CD tasks.
- `docs`: for changes in documentation.
- `chore`: for other changes that do not affect the code or tests (e.g., updating .gitignore, README.md).



```bash
# Add the files you have changed
git add .

# Create a commit with a proper message
git commit -m "feat(auth): implement user authentication"
```

Try to divide commits into atomic and independent parts. That is, do not add unrelated changes to the code in one commit.



### 4. Push Changes to Your Fork

After you have made one or more commits, push your branch to your remote fork on GitHub:

```bash
git push
```



### 5. Create a Pull Request (PR)

Once your branch with the changes is in your fork, you can create a Pull Request to propose your changes to the main repository.

1. Go to your fork's page on GitHub (https://github.com/YOUR-USERNAME/pysatl-mpest).
2. You will see a notification prompting you to create a Pull Request for your recently pushed branch. Click the **"Compare & pull request"** button.
3. Ensure the base repository is PySATL/pysatl-mpest with the main branch, and the head repository is your fork and your working branch.
4. Give your Pull Request a meaningful title (following the [Conventional Commits](https://www.google.com/url?sa=E&q=https%3A%2F%2Fwww.conventionalcommits.org%2F) standard) and add a detailed description of the changes you've made.

#### Rules for Your PR to be Merged

For your Pull Request to be merged, it must meet the following requirements configured in the repository:

- **CI/CD Checks Must Pass:** All automated status checks (tests, linters, etc.) configured in the  project must complete successfully. If any check fails, you will need to fix the issue and push new commits to your branch.
- **Mandatory Code Review:**
  - Your PR must receive at least **one approval** from another project member.
  - The PR author **cannot** approve their own work. The most recent set of changes must be approved by someone other than the person who pushed them.
  - If you push new commits to your branch after receiving an approval, that  approval will be automatically dismissed. The changes will need to be  reviewed again. This ensures that the latest version of the code is  always reviewed.
- **All Conversations Must Be Resolved:** All comments and conversations on the code left during the review must be marked as resolved before the PR can be merged.
- **No Force Pushing:** The main branch is protected from force pushes to ensure a stable and clean commit history.

Each pull request must be reviewed by one of the maintainers:

* Danil Totmyanin ([iraedeus](https://github.com/iraedeus))



