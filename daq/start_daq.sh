#!/bin/bash

# Start a tmux daq0 session with a daq window, if it does not exist
# https://unix.stackexchange.com/questions/443569/start-tmux-and-execute-a-set-of-commands-on-boot
if ! tmux has-session -t daq0:daq > /dev/null 2>&1; then
	echo "No tmux daq0:daq session running, starting..."
	tmux new-session -d -s daq0 -n daq
	# Wait 1 second for the shell started by tmux to begin, avoids printing the below command twice
	# https://stackoverflow.com/a/70203248
	sleep 1
else
	echo "Found tmux daq0:daq running, doing nothing."
fi

# start daq.py, if it is not running
if ! pgrep -fc "python -u daq.py" > /dev/null 2>&1; then
	echo "No python -u daq.py running, starting..."
	# start daq.py in daq0:daq
	tmux send-keys -t daq0:daq "cd /home/mepland/chance_of_showers/daq && poetry run python -u daq.py" Enter
else
	echo "Found python -u daq.py running, doing nothing"
fi
