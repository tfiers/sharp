# Generated with:
#   _SHARP_COMPLETE=source sharp > enable-autocompletion.sh
#
# Alternative file contents:
#   eval "$(_SHARP_COMPLETE=source sharp)"

_sharp_completion() {
    local IFS=$'
'
    COMPREPLY=( $( env COMP_WORDS="${COMP_WORDS[*]}" \
                   COMP_CWORD=$COMP_CWORD \
                   _SHARP_COMPLETE=complete $1 ) )
    return 0
}

_sharp_completionsetup() {
    local COMPLETION_OPTIONS=""
    local BASH_VERSION_ARR=(${BASH_VERSION//./ })
    # Only BASH version 4.4 and later have the nosort option.
    if [ ${BASH_VERSION_ARR[0]} -gt 4 ] || ([ ${BASH_VERSION_ARR[0]} -eq 4 ] && [ ${BASH_VERSION_ARR[1]} -ge 4 ]); then
        COMPLETION_OPTIONS="-o nosort"
    fi

    complete $COMPLETION_OPTIONS -F _sharp_completion sharp
}

_sharp_completionsetup;
