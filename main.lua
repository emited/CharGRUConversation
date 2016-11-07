
require 'optim'
--local model_utils = require 'model_utils'
local DataSetLoader = require 'DataSetLoader'


local opt = {
	filename = 'data/qa.csv',
	batchSize = 1000,
}

local loader = DataSetLoader.create(opt.filename, opt.batchSize)
print(loader:nextBatch())
print(loader:nextBatch())
print(loader:nextBatch())

