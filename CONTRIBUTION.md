# How to Contribute

This repo welcomes contributions and suggestions.

## Prerequisites:

- Read Code Of Conduct before making any contributions or suggestions.

## Branch Naming Convention

- Feature branches ```feature/<Issue Number>``` e.g. ```feature/12345``` - ***Issue Number*** is the number of the issue item or work item in the GitHub issues page or project board. The issue number should be 5 digits, ***For example: if your issue number is 7 then your feature branch will be feature/00007***.
- Hotfix branches ```hotfix/<Issue Number>``` e.g. ```hotfix/52679``` - ***Issue Number*** is the number of Bug/Issue Item in the GitHub project board. Hotfix branches are used for patch releases.

The standard process for working off the `develop` branch should go as follows:
1. `git fetch` in your current local branch to get latest remote branches list
2. `git checkout develop` to switch to `develop` branch
3. `git pull` to get the latest of `develop` branch
4. `git checkout -b branch_name` to create local branch for your task/issue
   * This is consistent with the Git Flow approach and with Azure DevOps integration requirements
5. Go through the motions of development and committing and pushing as before. Once completed, create a pull request into `develop` branch
6. When pull request is completed, repeat from the top to start a new local branch/new dev work item

## Testing

 * Always write tests for any parts of the code base you have touched. These tests should cover all possible edge cases (malformed input, guarding against nulls, infinite loops etc)