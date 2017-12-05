--[[
Copyright (c) 2014 Google Inc.
See LICENSE file for full terms of limited license.
]]

gd = require "gd"

require 'cutorch';
require 'nn';
require 'paths';
require 'cunn';

sl_net = torch.load('slnn-23.t7')
net = sl_net.model

print("@!#!@#!@#!@#!@#")
print(net)

function map_action_index(sl_action)
    print("$$$$$$$$$$$$$$$$")
    print(sl_action:exp())
    print("$$$$$$$$$$$$$$$$")

    local confidences, indices = torch.sort(sl_action, true)  -- true means sort in descending order
    print(indices[1])
    action_index = 1
    
    for i = 1, 15 do
        if indices[i] == 4 then
            action_index = 2
            break
        elseif indices[i] == 3 then
            action_index = 5
            break
        elseif indices[i] == 2 then
            action_index = 6
            break
        elseif indices[i] == 6 then
            action_index = 9
            break
        elseif indices[i] == 9 then
            action_index = 10
            break
        elseif indices[i] == 7 then
            action_index = 13
            break
        elseif indices[i] == 10 then
            action_index = 14
            break
        end
    end
    return action_index
end

if not dqn then
    require "initenv"
end

local cmd = torch.CmdLine()
cmd:text()
cmd:text('Train Agent in Environment:')
cmd:text()
cmd:text('Options:')

cmd:option('-framework', '', 'name of training framework')
cmd:option('-env', '', 'name of environment to use')
cmd:option('-game_path', '', 'path to environment file (ROM)')
cmd:option('-env_params', '', 'string of environment parameters')
cmd:option('-pool_frms', '',
           'string of frame pooling parameters (e.g.: size=2,type="max")')
cmd:option('-actrep', 1, 'how many times to repeat action')
cmd:option('-gameOverPenalty', 0, 'penalty for the game ending')
cmd:option('-random_starts', 0, 'play action 0 between 1 and random_starts ' ..
           'number of times at the start of each training episode')

cmd:option('-name', '', 'filename used for saving network and training history')
cmd:option('-network', '', 'reload pretrained network')
cmd:option('-agent', '', 'name of agent file to use')
cmd:option('-agent_params', '', 'string of agent parameters')
cmd:option('-seed', 1, 'fixed input seed for repeatable experiments')

cmd:option('-verbose', 2,
           'the higher the level, the more information is printed to screen')
cmd:option('-threads', 1, 'number of BLAS threads')
cmd:option('-gpu', -1, 'gpu flag')
cmd:option('-gif_file', '', 'GIF path to write session screens')
cmd:option('-csv_file', '', 'CSV path to write session data')

cmd:text()

local opt = cmd:parse(arg)

--- General setup.
local game_env, game_actions, agent, opt = setup(opt)

-- override print to always flush the output
local old_print = print
local print = function(...)
    old_print(...)
    io.flush()
end

-- file names from command line
local gif_filename = opt.gif_file

-- start a new game
local screen, reward, terminal = game_env:newGame()

-- compress screen to JPEG with 100% quality
local jpg = image.compressJPG(screen:squeeze(), 100)
-- create gd image from JPEG string
local im = gd.createFromJpegStr(jpg:storage():string())
-- convert truecolor to palette
im:trueColorToPalette(false, 256)

-- write GIF header, use global palette and infinite looping
im:gifAnimBegin(gif_filename, true, 0)
-- write first frame
im:gifAnimAdd(gif_filename, false, 0, 0, 7, gd.DISPOSAL_NONE)

-- remember the image and show it first
local previm = im
local win = nil
-- local win = image.display({image=screen})

print("Started playing...")
count = 0

-- play one episode (game)
while not terminal do
    -- if action was chosen randomly, Q-value is 0
    agent.bestq = 0
    
    -- choose the best action
    local action_index = agent:perceive(reward, screen, terminal, true, 0.05)

    sl_action = net:forward(screen)
    print("$$$$$$$$$$$$$$$$")
    print(sl_action:exp())
    print("$$$$$$$$$$$$$$$$")
    action_index = map_action_index(sl_action)

    local confidences, indices = torch.sort(sl_action, true)  -- true means sort in descending order
    print(indices[1])
    print(action_index)
    if last_act == action_index and last_act == 9 then
        count = count + 1
    end

    if count > 30 then 
        action_index = 1
        count = 0
    end
    -- action_index = 9
    last_act = action_index


    -- play game in test mode (episodes don't end when losing a life)
    screen, reward, terminal = game_env:step(game_actions[action_index], false)
    -- print(screen)

    -- display screen
    -- image.display({image=screen, win=win})

    -- create gd image from tensor
    jpg = image.compressJPG(screen:squeeze(), 100)
    im = gd.createFromJpegStr(jpg:storage():string())
    
    -- use palette from previous (first) image
    im:trueColorToPalette(false, 256)
    im:paletteCopy(previm)

    -- write new GIF frame, no local palette, starting from left-top, 7ms delay
    im:gifAnimAdd(gif_filename, false, 0, 0, 7, gd.DISPOSAL_NONE)
    -- remember previous screen for optimal compression
    previm = im

end

-- end GIF animation and close CSV file
gd.gifAnimEnd(gif_filename)

print("Finished playing, close window to exit!")


-- 0 == up
-- 1 == down
-- 2 == left
-- 3 == right
-- 4 == a
-- 5 == b
-- 6 == right + a
-- 7 == right + b
-- 8 == right + a + b
-- 9 == left + a
-- 10 == left + b
-- 11 == left + a + b
-- 12 == left + down
-- 13 == right + down
-- 14 == start
-- 15 == select

        -- 2 = a
        -- 5 = right    
        -- 6 = left
        -- 8 = a
        -- 9 = right a
        -- 10 = left a
        -- 11 = a
        -- 13 = b right
        -- 14 = b left

