def numDots(url):
	ret = 0
	for i in range(0, len(url)):
		if(url[i] == '.'):
			ret+=1
	return ret

def subDomainLevel(url):
	x = url.split('/')
	ret = 0
	for i in range(0, len(x[0])):
		if(x[0][i] == '.'):
			ret+=1
	return ret

def pathLevel(url):
	x = url.split('/')
	return len(x)-1

def urlLength(url):
	return len(url)

def numDash(url):
	return len(url.split('-'))-1

def numDashInHostName(url):
	x = url.split('/')
	x = x[0]
	return len(x.split('-'))-1

def atSymbol(url):
	if '@' in url:
		return 1
	else:
		return 0

def tildeSymbol(url):
	if '~' in url:
		return 1
	else:
		return 0

def numUnderscore(url):
	return len(url.split('_'))-1

def numPercent(url):
	return len(url.split('%'))-1

# def numQueryComponents(url):
# 	x = url.split('query')

def numAmpersand(url):
	return len(url.split('&'))-1

def numHash(url):
	return len(url.split('#'))-1

def numNumericChars(url):
	ret = 0
	for i in range(0, len(url)):
		if(url[i] >= '0' and url[i] <= '9'):
			ret+=1
	return ret

def isIPv4(s):
    try: return str(int(s)) == s and 0 <= int(s) <= 255
    except: return False

def isIPv6(s):
    if len(s) > 4:
        return False
    try : return int(s, 16) >= 0 and s[0] != '-'
    except:
        return False

def ipAddr(url):
	IP = url.split('/')[0]
	if IP.count(".") == 3 and all(isIPv4(i) for i in IP.split(".")):
		return 1
	if IP.count(":") == 7 and all(isIPv6(i) for i in IP.split(":")):
		return 1
	return 0

def hostnameLength(url):
	return len(url.split('/')[0])

def pathLength(url):
	return len(url) - hostnameLength(url)

def numSensitiveWords(url):
	sens = ['secure', 'account', 'webscr', 'login', 'ebayisapi', 'sign', 'banking', 'confirm']
	ret = 0
	for word in sens:
		if word in url:
			ret+=1
	return ret