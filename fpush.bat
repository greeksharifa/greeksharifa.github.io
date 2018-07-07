@echo off

git add .
git status

set str=
set /p str=enter commit message :

git commit -m "%str%"

git rebase -i HEAD~2

git push origin +master
