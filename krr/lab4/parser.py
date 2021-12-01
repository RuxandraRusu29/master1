class Parser(object):
    @staticmethod
    def parse(file: str):
        '''
        @param file: path to the input file
        :returns Bayesian network as a dictionary {node: [list of parents], ...}
        and the list of queries as [{"X": [list of vars], 
        "Y": [list of vars], "Z": [list of vars]}, ... ] where we want 
        to test the conditional independence of vars1 ⊥ vars2 | cond 
        '''
        bn = {}
        queries = []

        with open(file) as fin:
            # read the number of vars involved
            # and the number of queries
            N, M = [int(x) for x in next(fin).split()]
            
            # read the vars and their parents
            for i in range(N):
                line = next(fin).split()
                var, parents = line[0], line[1:]
                bn[var] = parents

            # read the queries
            for i in range(M):
                vars, cond = next(fin).split('|')

                # parse vars
                X, Y = vars.split(';')
                X = X.split()
                Y = Y.split()

                # parse cond
                Z = cond.split()

                queries.append({
                    "X": X,
                    "Y": Y,
                    "Z": Z
                })

            # read the answers
            for i in range(M):
                queries[i]["answer"] = next(fin).strip()

        return bn, queries

    @staticmethod
    def get_graph(bn: dict):
        '''
        @param bn: Bayesian netowrk obtained from parse
        :returns the graph as {node: [list of children], ...}
        '''
        graph = {}

        for node in bn:
            parents = bn[node]

            # this is for the leafs
            if node not in graph:
                graph[node] = []

            # for each parent add 
            # the edge parent->node
            for p in parents:
                if p not in graph:
                    graph[p] = []
                graph[p].append(node)

        return graph

class Graph:
  
    def __init__(self):
        self.graph = {}
  
    def addEdge(self, u, v):
        if u not in self.graph:
            self.graph[u] = [v]
        else:
            self.graph[u].append(v)
  
    def DFS(self, v, visited, u, path, query):
    
        visited.add(v)
        path.append(v)
        for neighbour in self.graph[v]:
            if neighbour not in visited:
                if neighbour == u:
                    path.append(u)
                    rez = check_path_open(path, query)
                    return rez
                else:
                    if self.DFS(neighbour, visited.copy(), u, path.copy(), query):
                        return True
                
        return False

def check_path_open(path, query):
    if len(path) < 3:
        return False
        
    for i in range(0, len(path) - 2):
        if path[i+1] in graph[path[i]] and path[i+2] in graph[path[i+1]]:
            if path[i+1] in query['Z']:
                return False
            else:
                return True
        if path[i] in graph[path[i+1]] and path[i+1] in graph[path[i+2]]:
            if path[i+1] in query['Z']:
                return False
            else:
                return True
                
        if path[i] in graph[path[i+1]] and path[i+2] in graph[path[i+1]]:
            if path[i+1] in query['Z']:
                return False
            else:
                return True
        if path[i+1] in graph[path[i]] and path[i+1] in graph[path[i+2]]:
            for desc in descendats[path[i+1]]:
                if desc in query['Z']:
                    return True
    return False
    
def compute_descendants(a, list_aux):
    
    list_aux.append(a)
    for neighbour in graph[a]:
        compute_descendants(neighbour, list_aux)

if __name__ == "__main__":
    from pprint import pprint
    
    # example usage
    bn, queries = Parser.parse("bn1")
    graph = Parser.get_graph(bn)
    
    print("Bayesian Network\n" + "-" * 50)
    pprint(bn)

    print("\nQueries\n" + "-" * 50)
    pprint(queries)

    print("\nGraph\n" + "-" * 50)
    pprint(graph)
    

    g = Graph()
    for i in graph:
        for j in graph[i]:
            g.addEdge(i, j)
            g.addEdge(j, i)
    print(g.graph)
    
    descendats = {}
    
    for a in graph:
        list_aux = []
        compute_descendants(a, list_aux)
        descendats[a] = list_aux
    
    print("Following is DFS from (starting from vertex 2)")
    
    for query in queries:
        rez = True
        print(query)
        for i in query['X']:
            for j in query['Y']:
                if g.DFS(i, set(), j, [], query):
                    rez = False
        print(rez)
    
  