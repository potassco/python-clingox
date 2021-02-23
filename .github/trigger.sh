#!/bin/bash

dev_branch=wip

function list() {
    curl \
      -X GET \
      -H "Accept: application/vnd.github.v3+json" \
      "https://api.github.com/repos/potassco/python-clingox/actions/workflows" \
      -d "{\"ref\":\"${dev_branch}\"}"
}

function dispatch() {
    token=$(grep -A1 workflow_dispatch ~/.tokens | tail -n 1)
    curl \
      -u "rkaminsk:$token" \
      -X POST \
      -H "Accept: application/vnd.github.v3+json" \
      "https://api.github.com/repos/potassco/python-clingox/actions/workflows/$1/dispatches" \
      -d "{\"ref\":\"${dev_branch}\"}"
}

case $1 in
    list)
        list
        ;;
    dev)
        # .github/workflows/ppa-dev.yml
        dispatch 5997696
        # .github/workflows/pipsource.yml
        dispatch 6065447
        # .github/workflows/conda-dev.yml
        dispatch 6066933
        ;;
    release)
        echo "implement me"
        ;;
    *)
        echo "usage: trigger {list,dev,release}"
        ;;
esac
