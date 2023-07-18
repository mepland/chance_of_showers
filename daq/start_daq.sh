#!/bin/bash

# Start a tmux daq session with a daq window, if it does not exist
# https://unix.stackexchange.com/questions/443569/start-tmux-and-execute-a-set-of-commands-on-boot
if ! tmux has-session -t daq > /dev/null 2>&1; then
	echo "No tmux daq session running, starting..."
	tmux new-session -d -s daq
	# Wait 1 second for the shell started by tmux to begin, avoids printing later commands twice
	# https://stackoverflow.com/a/70203248
	sleep 1
else
	echo "Found tmux daq session running, doing nothing."
fi

# now start window
if ! tmux has-session -t daq:daq > /dev/null 2>&1; then
	echo "No tmux daq:daq session:window running, starting window..."
	tmux new-window -d -n daq
	# move daq window to index 0, swapping current 0 to end of list if needed
	tmux swap-window -d -s daq -t 0
	sleep 1
else
	echo "Found tmux daq:daq session:window running, doing nothing."
fi

# start daq.py, if it is not running
if ! pgrep -fc "python -u daq.py" > /dev/null 2>&1; then
	echo "No python -u daq.py running, starting..."
	# start daq.py in daq:daq
	tmux send-keys -t daq:daq "cd /home/mepland/chance_of_showers/daq && poetry run python -u daq.py" Enter
else
	echo "Found python -u daq.py running, doing nothing"
fi