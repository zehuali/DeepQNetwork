#!/bin/bash

if [ -z "$1" ]
  then echo "Please provide the name of the game, e.g.  ./watch_pretrained breakout"; exit 0
fi

if [ -z "$2" ]
  then echo "Please provide the pretrained network file, e.g.  ./watch_pretrained breakout DQN3_0_1_breakout_FULL_Y.t7"; exit 0
fi

ENV=$1
NETWORK=$2
FRAMEWORK="neswrap"

game_path=$PWD"/roms/"
env_params="useRGB=true"
agent="NeuralQLearner"
n_replay=4
netfile="\"convnet_nes\""
update_freq=4
actrep=8
discount=0.99
seed=1
learn_start=5000
pool_frms_type="\"max\""
pool_frms_size=1
initial_priority="false"
replay_memory=100 # This doesn't matter for testing...
eps_end=0.1
eps_endt=500000
lr=0.01
agent_type="DQN3_0_1"
preproc_net="\"net_downsample_2x_full_y\""
agent_name=$agent_type"_"$1"_FULL_Y"
state_dim=7056
ncols=1
agent_params="lr="$lr",ep=1,ep_end="$eps_end",ep_endt="$eps_endt",discount="$discount",hist_len=4,learn_start="$learn_start",replay_memory="$replay_memory",update_freq="$update_freq",n_replay="$n_replay",network="$netfile",preproc="$preproc_net",state_dim="$state_dim",minibatch_size=256,rescale_r=1,ncols="$ncols",bufferSize=1024,valid_size=1000,target_q=10000,clip_delta=1,min_reward=-10000,max_reward=10000"
gif_file="../gifs/$ENV.gif"
gpu=0
random_starts=0
pool_frms="type="$pool_frms_type",size="$pool_frms_size
num_threads=8

args="-framework $FRAMEWORK -game_path $game_path -name $agent_name -env $ENV -env_params $env_params -agent $agent -agent_params $agent_params -actrep $actrep -gpu $gpu -random_starts $random_starts -pool_frms $pool_frms -seed $seed -threads $num_threads -network $NETWORK -gif_file $gif_file"
echo $args

cd dqn
# ../torch/bin/qlua test_agent.lua $args
../torch/bin/qlua test_sl.lua $args
