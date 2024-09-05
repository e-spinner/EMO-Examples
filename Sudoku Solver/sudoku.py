from random import sample
import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.callback import Callback
from IPython.display import clear_output
import numpy as np
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.termination import Termination
from pymoo.core.mutation import Mutation
import time



def generateBoard( group_size, x, y ) :
        side_size = group_size * group_size
        
        # define pattern to select valid numbers. Ensures that numbers are placed in nonconflicting cells
        def index_pattern( row, col ) : return ( group_size * ( row % group_size ) + row // group_size + col ) % side_size
        
        # shuffle  possible values to randomize solution
        def shuffle_group( ) : return sample( range( group_size ), len( range( group_size ) ) )
        
        # generate random indexing for rows and columns in solution
        rows = [ index * group_size + row for index in shuffle_group() for row in shuffle_group() ]
        cols = [ index * group_size + col for index in shuffle_group() for col in shuffle_group() ]
        
        # generate random set of numbers to arrange into rows and cols
        nums = sample( range( 1, side_size + 1 ), len( range( 1, side_size + 1) ) )
        
        # arrange nums according to index pattern and generated rows and cols order
        board = [ [ nums[ index_pattern( row, col ) ] for col in cols ] for row in rows ]
        
        solution = np.array( board )
        
        # set a ratio of the board to zero randomly
        num_cells = side_size * side_size
        num_empty = num_cells * x // y
        for delete in sample( range( num_cells ), num_empty ) :
            board[ delete // side_size ][ delete % side_size ] = 0
            
        return ( np.array( board ), solution )

def compare( input, output, solution ) :
    red = '\x1b[1;91m'      
    blue = '\x1b[1;92m'
    white = '\x1b[1;97m'
    gray = '\x1b[1;37m'
    green = '\x1b[1;94m'
    g1 = np.sum( output, axis=1 ) == 45
    g2 = np.sum( output, axis=0 ) == 45
    g3 = np.sum( np.array( [ output[ row:row+3, col:col+3 ] for row in range( 0, output.shape[0], 3 ) \
                                                        for col in range( 0, output.shape[1], 3 ) ] ), axis=(1,2) ) == 45
    g4 = np.array( [ np.sum( output, axis=(0, 1) ) for _ in range( 0, 9 ) ] ) == 405
    g5 = np.array( [ np.sum( np.where( output == num, 1, 0 ) ) for num in range( 1, 10 ) ] ) == 9
    
    check = '\x1b[1;92m✔ '
    cross = '\x1b[1;91m✘ '
    print()
    print( white, ' Input Board:             Solution:                 Output Board:           Constraints: [%s]' % np.sum( np.concatenate( [ g1, g2, g3, g4, g5 ] ) == False ) )
    print( green, '  1 2 3  4 5 6  7 8 9      1 2 3  4 5 6  7 8 9       1 2 3  4 5 6  7 8 9', end='' )
    print( '     ', end='' )
    for col in range( 0 , 9 ) :
        if col == 3 or col == 6 : print( ' ', end='' )
        if g2[col] : print( check, end='' )
        else : print( cross, end='' )
    print()
    for row in range( 0, 9 ) :
        
        # input
        print( green, row + 1, gray, end='' )
        for col in range( 0, 9 ) :
            if col == 3 or col == 6 : print( ' ', end='' )
            print( '%s ' % input[row][col] if input[row][col] != 0 else '\x1b[1;94m· \x1b[1;37m', end='' )
        print( '  ', end='' )
        
        # solution
        print( green, row + 1, gray, end='' )
        for col in range( 0, 9 ) :
            if col == 3 or col == 6 : print( ' ', end='' )
            color = gray if solution[row][col] == input[row][col] else white
            print( color, end='' )
            print( '%s ' % ( solution[row][col] if color == white else '·' ), end='' )
        print( '   ', end='' )
        
        # output
        print( green, row + 1, white, end='' )
        for col in range( 0, 9 ) :
            if col == 3 or col == 6 : print( ' ', end='' )
            color = gray if solution[row][col] == input[row][col] else blue if solution[row][col] == output[row][col] else red
            print( color, end='' )
            print( '%s ' % output[row][col], end='' )
        print( '  ', end='' )
                
        # constriants
        if g1[row] : print( check, end='' )
        else : print( cross, end='' )
        
        if row == 1 or row == 4 or row == 7 : 
            for cell in range( 0, 3 ) :
                print( '\x1b[1;97m· ', end='' )
                if g3[ cell + ( row - 1 ) ] : print( check, end='' )
                else : print( cross, end='' )
                if cell == 2 : print( '\x1b[1;97m· ', end='' )
                else : print( '\x1b[1;97m·  ', end='' )
        elif row == 8: print( '\x1b[1;97m· · ·  · · ·  · · %s' % ( check if g4[0] else cross ), end='' )
        else : print( '\x1b[1;97m· · ·  · · ·  · · · ', end='' )
        
        print( '\x1b[1;97m %s: ' % ( row + 1 ), end='' )
        if g5[row] : print( check, end='' )
        else : print( cross, end='' )
        
        print()
        if row == 2 or row == 5 : print()
    print()

def evaluate( output ) :
        g1 = np.sum( output, axis=1 ) == 45
        g2 = np.sum( output, axis=0 ) == 45
        g3 = np.sum( np.array( [ output[ row:row+3, col:col+3 ] for row in range( 0, output.shape[0], 3 ) \
                                                            for col in range( 0, output.shape[1], 3 ) ] ), axis=(1,2) ) == 45
        g4 = np.array( [ np.sum( output, axis=(0, 1) ) for _ in range( 0, 9 ) ] ) == 405
        g5 = np.array( [ np.sum( np.where( output == num, 1, 0 ) ) for num in range( 1, 10 ) ] ) == 9
        return np.concatenate( [ g1, g2, g3, g4, g5 ] ) 
    
def check( board ) :
    return np.array( [ solution.reshape( ( 81 ) ) for _ in range( 0, board.shape[0] ) ] )   

solution = []
input_board = []

class SudokuPuzzle() :
    def __init__( self, x, y ) :
        global solution
        self.input_board, solution = generateBoard( 3, x, y )
        global input_board
        input_board = self.input_board
        

    class NSGA2_end( NSGA2 ) :
        def __init__(self,
                    pop_size,
                    sampling,
                    crossover,
                    mutation,
                    **kwargs) :
            super().__init__(
                    pop_size=pop_size,
                    sampling=sampling,
                    crossover=crossover,
                    mutation=mutation,
                    **kwargs)
            self.end = False

    class DisplayBestBoard ( Callback ):

        def __init__( self ) -> None:
            super().__init__()

        def notify( self, algorithm ):
            
            output = algorithm.opt.get( "X" )        
            output = output if output.size <= 81 else output[ 1, :81 ]
            
            end = np.all( evaluate( output.reshape( ( 9, 9 ) ) ) )
            
            if algorithm.n_gen % 1 == 0 or end :
                
                time.sleep(1)
                
                clear_output()
                print( '\x1b[1;97m Generation: %s'% algorithm.n_gen )
                compare( input_board, output.reshape( ( 9,9 ) ), solution )
            
            algorithm.end = end
            
    class CheckTermination ( Callback ):

        def __init__( self ) -> None:
            super().__init__()

        def notify( self, algorithm ):
            
            output = algorithm.opt.get( "X" )        
            output = output if output.size <= 81 else output[ 1, :81 ]
            
            end = np.all( evaluate( output.reshape( ( 9, 9 ) ) ) )
                            
            algorithm.end = end
            
    class End( Termination ):

        def __init__(self) -> None:
            super().__init__()

        def _update(self, algorithm):
            end = algorithm.end

            return end

    class SelectiveMutation( Mutation ):

        def _do( self, problem, X, **kwargs ):  
           
            Xp = np.where( X == check(X), X, np.random.randint(1, 9) )
            return Xp
        
    class SudokuSolver( ElementwiseProblem ) :
        def __init__( self, input_board, **kwargs ) :
            lower = np.where( input_board != 0, input_board, 1 ).reshape( ( 81 ) )
            upper = np.where( input_board != 0, input_board, 9 ).reshape( ( 81 ) )
            
            super().__init__( n_var=81, n_obj=1, n_ieq_constr=45+81, xl=lower, xu=upper )
            
        def _evaluate( self, X, out, *args, **kwargs ) :
            
            # X is ( ( 81 ) ) --> ( ( 9, 9 ) )        
            X = X.reshape( ( 9, 9 ) )
            
            # Each row must equal to 45. 
            g1 = np.sum( X, axis=1 ) == 45

            # Each col must equal to 45. 
            g2 = np.sum( X, axis=0 ) == 45

            # Each block must equal to 45. 
            g3 = np.sum( np.array( [ X[ row:row+3, col:col+3 ] for row in range( 0, X.shape[0], 3 ) \
                                                            for col in range( 0, X.shape[1], 3 ) ] ), axis=(1,2) ) == 45
            
            # The Board must equal to 405. 
            g4 = np.array( [ np.sum( X, axis=(0, 1) ) for _ in range( 0, 9 ) ] ) == 405

            # There must be nine of each number
            g5 = np.array( [ np.sum( np.where( X == num, 1, 0 ) ) for num in range( 1, 10 ) ] ) == 9
                            
            # print( 'g1: %s\ng2: %s\ng3: %s\ng4: %s\ng5: %s' % ( g1, g2, g3, g4, g5 ) )
            
            g6 =( X == solution ).reshape( ( 81 ) )

            out["G"] = np.concatenate( [ g1<=0, g2<=0, g3<=0, g4<=0, g5<=0, g6<=0 ], axis=0 )
            out["F"] = 0
            
