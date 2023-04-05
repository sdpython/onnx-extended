Useful commands on Linux
========================

Git
+++

* clone: `git clone <repo address.git>`
* create a new branch: `git checkout -b <new_branch>`
* add a remote repository: `git remote add <remote name> <remote address>`
* merge modification: `git pull <remote name> <remote branch>`
* add modified files: `git add <files or folder>`
* commit added files: `git commit -m "commit message"`
* push modifications to the remote repository: `git push`
* remove all current modifications: `git reset --hard`
* show modified filed: `git status`

Retrieve information about the CPU
++++++++++++++++++++++++++++++++++

::

    cat /proc/cpuinfo
    lscpu
