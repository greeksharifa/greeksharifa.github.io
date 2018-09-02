@echo off

git add .
git status

git commit -m "If you need, edit me!"

git rebase -i HEAD~2

git push origin +master
