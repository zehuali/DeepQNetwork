-- nn.Sequential {
--     [input -> (1) -> (2) -> (3) -> (4) -> (5) -> (6) -> (7) -> (8) -> (9) -> (10) -> (11) -> output]
--     (1): nn.Reshape(4x84x84)
--     (2): nn.SpatialConvolution(4 -> 32, 8x8, 4,4, 1,1)
--     (3): nn.Rectifier
--     (4): nn.SpatialConvolution(32 -> 64, 4x4, 2,2)
--     (5): nn.Rectifier
--     (6): nn.SpatialConvolution(64 -> 128, 3x3)
--     (7): nn.Rectifier
--     (8): nn.Reshape(6272)
--     (9): nn.Linear(6272 -> 512)
--     (10): nn.Rectifier
--     (11): nn.Linear(512 -> 15)
--   }

require 'cutorch';
require 'nn';
require 'paths';
require 'cunn';

trainset = torch.load('train_set-1.t7')
-- testset = torch.load('cifar10-test.t7')

setmetatable(trainset, 
{__index = function(t, i) 
                return {t.data[i], t.label[i]} 
            end}
);
trainset.data = trainset.data:double() -- convert the data from a ByteTensor to a DoubleTensor.

function trainset:size() 
return self.data:size(1) 
end

redChannel = trainset.data[{ {}, {1}, {}, {}  }] -- this picks {all images, 1st channel, all vertical pixels, all horizontal pixels}
print(#redChannel)

mean = {} -- store the mean, to normalize the test set in the future
stdv  = {} -- store the standard-deviation for the future
for i=1,3 do -- over each image channel
    mean[i] = trainset.data[{ {}, {i}, {}, {}  }]:mean() -- mean estimation
    print('Channel ' .. i .. ', Mean: ' .. mean[i])
    trainset.data[{ {}, {i}, {}, {}  }]:add(-mean[i]) -- mean subtraction
    
    stdv[i] = trainset.data[{ {}, {i}, {}, {}  }]:std() -- std estimation
    print('Channel ' .. i .. ', Standard Deviation: ' .. stdv[i])
    trainset.data[{ {}, {i}, {}, {}  }]:div(stdv[i]) -- std scaling
end


net = nn.Sequential()
net:add(nn.SpatialConvolution(3, 32, 8, 8, 4, 4, 1, 1)) -- 3 input image channels, 6 output channels, 5x5 convolution kernel
net:add(nn.ReLU())                       -- non-linearity 
net:add(nn.SpatialMaxPooling(2,2,2,2))     -- A max-pooling operation that looks at 2x2 windows and finds the max.
net:add(nn.SpatialConvolution(32, 64, 4, 4, 2, 2))
net:add(nn.ReLU())                       -- non-linearity 
net:add(nn.SpatialMaxPooling(2,2,2,2))
net:add(nn.SpatialConvolution(64, 128, 3, 3))
net:add(nn.View(128*4*5))                    -- reshapes from a 3D tensor of 16x5x5 into 1D tensor of 16*5*5
net:add(nn.Linear(128*4*5, 512))             -- fully connected layer (matrix multiplication between input and weights)
net:add(nn.ReLU())                       -- non-linearity 
net:add(nn.Linear(512, 16))                -- 10 is the number of outputs of the network (in this case, 10 digits)
net:add(nn.LogSoftMax())                     -- converts the output to a log-probability. Useful for classification problems


criterion = nn.ClassNLLCriterion()

net = net:cuda()
criterion = criterion:cuda()
trainset.data = trainset.data:cuda()
trainset.label = trainset.label:cuda()

trainer = nn.StochasticGradient(net, criterion)
trainer.learningRate = 0.001
trainer.maxIteration = 50 -- just do 5 epochs of training.

    
start = os.time(os.date("!*t"))
trainer:train(trainset)
print("time cost", os.time(os.date("!*t")) - start)

torch.save('slnn-1.t7', {model = net})