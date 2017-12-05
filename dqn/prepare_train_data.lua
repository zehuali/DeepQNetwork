require "image";

labelfile = 'train_data/1/labels.lua'
train_set_file_name = "train_set-1.t7"
size = 1000

lines = {}
for line in io.lines(labelfile) do 
    lines[#lines + 1] = tonumber(line)
end

print(lines)
-- size = #lines
total_size = #lines

turns = math.floor(total_size / size)

for i = 1, turns do



    imagesAll = torch.Tensor(size + 1,3,224,256)
    local labelsAll = torch.Tensor(size + 1)

    print(size)
    print(type(imagesAll))

    for f=0,size do
        -- print(f)
        ff = f + 1 + (i * 1000)
        imagesAll[f+1] = image.load('train_data/1/snaps/mario-'..ff..'.png') 
        labelsAll[f+1] = lines[ff] -- 2 = background
    end


    -- create train set:
    trainData = {
    data = imagesAll,
    label = labelsAll,
    size = function() return trsize end
    }

end

-- print(trainData)
torch.save(train_set_file_name, trainData)