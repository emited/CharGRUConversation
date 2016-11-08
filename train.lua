
require 'optim'
local gru = require 'gru'
local model_utils = require 'model_utils'
local DataSetLoader = require 'DataSetLoader'


local opt = {
	filename = 'data/qa.csv',
	batchSize = 1000,
	maxEpochs = 10,
	optimName = 'adam',
	optimState = {learningRate = 1e-3},
	seed = 123,
	saveEvery = 1000,
	saveFile = 'gru_model'
	printEvery = 1,
}

torch.manualSeed(opt.seed)

local protos = {}
protos.encodeGRU = gru.GRU()
protos.decodeGRU = gru.GRU()
protos.softmax = nn.Sequential():add(nn.Linear()):add(nn.SoftMax())
protos.criterion = nn.CrossEntropyCriterion()

local params, gradParams = model_utils.combine_all_parameters(
	protos.encodeGRU,
	protos.decodeGRU,
	protos.softmax)
params:uniform(-.08, .08)

local clones = {}
for name, proto in pairs(protos) do
	clones[name] = model_utils.clone_many_times(proto, )
end

local loader = DataSetLoader.create(opt.filename, opt.batchSize)




function feval(new_params)
	if new_params ~= params then
		params:copy(new_params)
	end
	gradParams:zero()
	local x, y = loader:nextBatch()
	grad_params:clamp(-5, 5)
	return loss, gradParams
end


local lossPlot = {}
local gradNormPlot = {}
for i = 1, iterations do

	local _, loss = optim[opt.optimName](feval, params, opt.optimState)

	--printing and evaluating
	if i % opt.printEvery == 0 then
        print(string.format("iteration %4d, loss = %6.8f, loss/seq_len = %6.8f, gradnorm = %6.4e", i, loss[1], loss[1] / opt.seq_length, gradParams:norm()))
	end
	if i % opt.saveEvery == 0 then
		torch.save(opt.saveFile..'_epoch_'..i..'.t7', protos)
	end
end