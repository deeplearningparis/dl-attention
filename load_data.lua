-- Dictionary
local char2id = {}
local id2char = {}

local maxIndex = 1
local function getCharId(char)
   if not char2id[char] then
      char2id[char] = maxIndex
      id2char[maxIndex] = char
      maxIndex = maxIndex + 1
   end
   return char2id[char]
end

-- space is used for padding and for decoder first char
getCharId(' ')

-- Data set iterator
local function dataSet(path)
   local it = io.lines(path)
   return function()
      local line = it()
      if not line then return nil end
      local source = {}
      local target = {}
      local first = true
      -- Read chars
      for i=1, #line do
         local char = line:sub(i, i)
         if first then
            first = char ~= '|'
            if first then table.insert(source, char) end
         else
            assert(char ~= '|')
            table.insert(target, char)
         end
      end
      return source, target
   end
end

-- Load dataset
local function loadDataset(path)
   -- 1st path
   -- count lines, source max length and target max length
   local maxSourceLength = 0
   local maxTargetLength = 0
   local lines = 0
   for source, target in dataSet(path) do
      maxSourceLength = math.max(#source, maxSourceLength)
      maxTargetLength = math.max(#target, maxTargetLength)
      lines = lines + 1
   end

   local sources = torch.Tensor(lines, maxSourceLength):fill(1)
   local targets = torch.Tensor(lines, maxTargetLength):fill(1)
   local i=1
   for source, target in dataSet(path) do
      for j=1, #source do
         sources[i][j] = getCharId(source[j])
      end
      for j=1, #target do
         targets[i][j] = getCharId(target[j])
      end
      i = i+1
   end

   return sources, targets
end

local sources, targets = loadDataset(sys.fpath()..'/data_numbers.csv')
local dict = {char2id = char2id, id2char = id2char}

-- sanity check
assert(#dict.id2char, 12)
assert(sources:size(1) == 50000)
assert(sources:size(2) == 7)
assert(targets:size(1) == 50000)
assert(targets:size(2) == 4)

-- invert first 2 lines and verify them
function check(data, i, value)
   local s = {}
   for j=1, data:size(2) do table.insert(s, id2char[data[i][j]]) end
   assert(table.concat(s) == value)
end
check(sources, 1, '97+2   ')
check(sources, 2, '8+269  ')
check(targets, 1, '99  ')
check(targets, 2, '277 ')

--print(dict)

return sources, targets, dict
