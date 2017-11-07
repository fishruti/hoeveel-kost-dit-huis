from numpy import mean, std

def grouping(col, data):
    unique = data[col].unique()
    pairs = dict()
    for thing in unique:
        means = mean(train[train['Neighborhood'] == thing]['SalePrice'])
        stds = std(train[train['Neighborhood'] == thing]['SalePrice'])
        pairs.update({thing: (means, stds)})
    grouping = dict()
    for key1 in pairs.keys():
        for key2 in pairs.keys():
            if pairs[key1][0] < pairs[key2][0] + 10000 and pairs[key1][0] > pairs[key2][0] - 10000 and key1 != key2:
                data[col] = data[col].replace(to_replace = [key2], value = [key1])
                grouping.update({key1: key2})
        
    return grouping

def assign_grouping(col, data, grouping):
    for x in data[col]:
        for key, value in grouping.items():
            if x in value:
                data[col] = data[col].replace(to_replace = [x], value = [key])
    return data[col]