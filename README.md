# 🧠 Machine Learning  Checklist (No Math Version)

A flexible roadmap for JOSEPH IT student to become ML experts through coding,
tools, and projects.\
Check items as I learn them:

------------------------------------------------------------------------

## 1. Core Programming & Tools

-   [*] Learn **Python**
    -   [*] Variables, loops, functions
    -   [*] Object-Oriented Programming (OOP)
    -   [*] File handling (CSV, JSON, text files)
    -   📚 Resource: [Python for Everybody (Coursera Free)](https://www.coursera.org/specializations/python)
    
-   [*] Master ML Libraries
    -   [*] NumPy (ONE WEEK)
                # 📌 NumPy Checklist for Machine Learning
                ## 🔹 Array Basics
                - [*] Create arrays: `np.array`, `np.zeros`, `np.ones`, `np.arange`, `np.linspace`, `np.eye`
                        - np.eye()
                            - Takes one 
                - [*] Inspect properties: `.shape`, `.ndim`, `.size`, `.dtype`
                    '''
                        - .size ->  Returns the total number of elements in an array.
                        - .ndim ->  Returns the number of dimensions(axes) in a array.
                        - .shape -> Returns a turple representing the shape of the array (Rows,Columns).
                        - .dtype -> Returns the data type of the array.
                    '''

                ## 🔹 Reshaping
                - [*] Change shape: `.reshape()`, `.resize()`
                        '''
                            - .reshape()
                                - Creates a new view of the array with the new shape
                                - Does not change the original
                                - The total numbe of elements remain the Same
                            
                            - .resize()
                                - Modifies the array in place
                                - If the new size is bigger it fills with zeros
                                - If the new size is smaller it truncates
                        '''
                - [*] Flatten: `.ravel()`, `.flatten()`
                        '''
                            - .ravel()
                                - Returns a flatten 1D of the array if possible
                                - Faster and memory efficient
                                - #View
                            
                            - .flatten()
                                - Always return a copy of the array
                                - Uses more memory
                                - Changes do not affect the original (#Copy)
                        '''

                ## 🔹 Indexing & Slicing
                - [*] Basic slicing (1D & 2D)
                - [*] Boolean indexing
                        '''
                            - d = np.array([[1, 2, 3, 4, 5, 6],[7,8,9,10,11,12]])
                            - d[d % 2 == 0] -- Conditional indexing
                             -> array([ 2,  4,  6,  8, 10, 12])
                        '''
                - [*] Fancy indexing (index lists/arrays)
                        '''
                            - d = np.array([[1, 2, 3, 4, 5, 6],[7,8,9,10,11,12]])
                            - d[1,:4:2]
                            - -> array([7, 9])
                        '''

                ## 🔹 Operations
                - [*] Element-wise math: `+`, `-`, `*`, `/`, `**`
                - [*] Universal functions: `np.exp`, `np.log`, `np.sqrt`, `np.sin`, `np.cos`
                        '''
                        - np.exp()
                            - Computes the exponential (e) in an array -> np.exp(x) = e^x
                            - np.exp([1,2,3,4])
                                - -> array([ 2.71828183,  7.3890561 , 20.08553692, 54.59815003])
                                
                        - np.log()
                            - Computes the basic logarithm of each elements in an array
                            - np.log([1,2,3,4])
                                - array([0.        , 0.69314718, 1.09861229, 1.38629436])
                        
                        - np.sqrt()
                            - Computes the square root of each Element in an array
                            - np.sqrt([25,64,49,81])
                                - -> array([5., 8., 7., 9.])
                                
                        - np.sin()
                            - Computes the sine of each element in an array.
                            - inputs is in radians NOT in degrees; Radians are the stadard unit in Programming and Mathematics trigonometry
                            - p = np.sin(np.deg2rad([25,64,49,81]))
                                - -> array([0.42261826, 0.89879405, 0.75470958, 0.98768834])
                        '''
                - [*] Aggregations: `np.sum`, `np.mean`, `np.median`, `np.std`, `np.var`, `np.min`, `np.max`
                        '''
                        - np.sum()
                            - Computes the sum of array elements over a given axis
                            - np.sum(arr) -> Sum of all elements
                            - np.sum(arr,axis=0) -> Sum along columns
                            - np.sum(arr, axis=1) -> Sum along Rows
                            
                        - np.mean()
                            - Computes the arithmetic mean (average) of array elements
                            
                                        sum of elements
                            -    mean = ---------------
                                        number of elements
                            
                            - np.mean([1,2,3,4,5])
                            - np.mean(arr) -> Mean of all elements
                            - np.mean(arr,axis=0) -> Mean of columns
                            - np.mean(arr,axis=1) -> Mean of rows
                        
                        - np.var()
                            - Computes the variance of elements
                                - Variance measures how far values spread out from the mean
                            - np.var(arr) -> variance of all elements
                            - np.var(arr,axis=0) -> variance of all elements down columns
                            - np.var(arr,axis=1) -> Variance across rows
                        
                        - np.max()
                            - Returns the maximum value in a array
                            - np.max(arr) -> Largest overall
                            - np.max(arr,axis=0) -> Largest in a columns
                            - np.max(arr,axis=1) -> Largest in a Row
                        
                        - np.std()
                            - Computes the stardard deviation of array elements
                            - stardard deviation = square root of variance
                            - np.std(arr) -> Stardard deviation of all element
                            - np.std(arr,axis=0) -> Stardard deviation of all element in a columns
                            - np.std(arr,axis=1) -> Stardard deviation of all element in a row
                        -
                        '''
                        
                - [*] Index-based aggregations: `np.argmin`, `np.argmax`
                        - np.argmin()
                            - Returns the index of the smallest element
                            - Works with Index in 2-D dataset
                        
                        - np.argmax()
                            - Returns the largest element
                            - Works with Index in 2-D dataset
                            
                ## 🔹 Linear Algebra
                - [*] Dot product & matrix multiplication: `np.dot`, `np.matmul`, `@`
                        - np.dot()
                            - Performs a dot product/ matrix multiplication depending on input shape
                            - c = np.array([1,2])
                            - d = np.array([[3,4,5],[6,7,5]])
                            - np.dot(c,d)
                            - -> array([15, 18, 15])
                        
                        - `@`
                            -It is a matrix multiplication operator
                        
                        - np.matmul
                            - Performs matrix multiplication
                        
                            
                - [*] Transpose: `.T {Transpose}` 
                            - It swaps the rows and columns of an array
                                `(m,n) ---->> (n,m)`
                                
                                A = np.array([[1,2,3,4],      ---> A.T --------------------->  array([[1, 4],
                                              [4,5,6,7]])  A.shape(2,4)                               [2, 5], A.T.shape(4,2)
                                                                                                      [3, 6],
                                                                                                      [4, 7]])                  
                                              
                                              
                                
                - [*] Determinant: `np.linalg.det`
                        - np.linalg.det()
                            - Computes the Determinant of a square Matrix
                - [*] Inverse: `np.linalg.inv`
                        - Computes the inverse of s square matrix
                        - Works if the matrix is non-singular (det != 0)
                - [*] Eigenvalues & vectors: `np.linalg.eig`
                        - Computes the Eigenvalues and Eigenvectors of a square matrix
                        
                ## 🔹 Broadcasting
                - [*] Understand broadcasting rules & usage
                        - When performing an operation between arrays of different shapes, Numpy tries to "strech" the smaller array so the shapees become compatible
                        `RULE 1`
                            - Align shapes from the rightmost dimensions
                            - Compare shapes element by element from right to left
                            - Two dimensions are compatible if: 
                                    - They are equal OR
                                    - One of the is 1
                        `RULE 2`
                            - If dimensions are not compatible error.
                            
                        `Example 1 = Vector + Scalar`
                            - a = np.array([1,2,3]) # shape(3,)
                            - b = 5 # Scalar
                            - a = b
                            - answ ---> [6 7 8] -- Scalar is broadcast to shape 3,
                            
                        `Example 2 = Matrix = Vector`
                            - a = np.array([[1,2,3],[4,5,6]]) # shape (2,3)
                            - b = np.array([10,20,30]) # shape (3,)
                            - a + b
                            - answ = [[11,22,33],[14,25,36]] # Vector (3,) is broadcast across rows of (2,3)
                        
                        `Example 3 = Column Vector + Row Vector`
                            - x = np.array([[1],
                                            [2],
                                            [3]]) # Shape (3,1)
                            
                            - y = np.array([10,20,30]) # shape(3,)
                            - x + y
                            - Output
                                [[11 21 31]
                                 [12 22 32]
                                 [13 23 33]]
                        `Example 4 = Incompatible Shapes`
                            - A = np.array([[1,2,3],
                                            [4,5,6]])
                            - b = np.array([1,2])
                            - A + b
                            - Output = 'ValueError - Not compatible'

                ## 🔹 Random Numbers
                - [*] Uniform & normal: `np.random.rand`, `np.random.randn`
                        - np.random.rand --> Generates random numbers from a Uniform distribution in (0,1).
                            - Always non-negative
                            - np.random-rand(d0,d1,....dn) #Dimensions
                            
                        - np.random.randn --> Generates random numbers froma a standard normal distribution.
                            - Values can be negative or positive
                            
                - [*] Integers: `np.random.randint`
                - [*] Reproducibility: `np.random.seed`

                ## 🔹 Combining & Splitting
                - [*] Stack arrays: `np.concatenate`, `np.vstack`, `np.hstack`
                        - np.vstack()
                            - Stack numpy arrays on top of each other vertically
                            - arr1 = np.array([1,2,3,4])
                            - arr2 = np.array([4,5,6,7])
                            - np.vstack(arr1,arr2)
                            - --> array([[1, 2, 3, 4],
                                         [4, 5, 6, 7]])
                            
                        - np.hstack()
                            - Joing numpy arrays together horizontally
                            - arr1 = np.array([1,2,3,4])
                            - arr2 = np.array([4,5,6,7])
                            - np.hstack([arr1,arr2])
                            - ---> array([1, 2, 3, 4, 4, 5, 6, 7])
                            
                - [*] Split arrays: `np.split`, `np.vsplit`, `np.hsplit`
                        - np.split()
                            - Splits an array into multiple sub-arrays
                            - You must rell Numpy how many splits or the indeces where to split
                                `Example`
                                    - a = np.arange(9)
                                    - np.split(a,3)
                                    - Output --> array([0,1,2],array([3,4,5],array([6,7,8])))
                            
                        - np.vsplit()
                            - Split along rows(Vertical direction)
                            - Works only in 2-D or higher
                            - n3 = np.array([[1,2,3,4],
                                             [5,6,7,8],
                                             [9,10,11,12],
                                             [13,14,15,16]])
                            - np.vsplit(n3,4)
                            - Output
                                    [array([[1, 2, 3, 4]]),
                                     array([[5, 6, 7, 8]]),
                                     array([[ 9, 10, 11, 12]]),
                                     array([[13, 14, 15, 16]])]
                                                            
                        - np.hsplit()
                            - Split along columns
                            - np.hsplit(n3,2)
                            - Output
                                

                ## 🔹 Copies & Views
                - [*] Shallow copy: `.view()`
                        - Create a new object that looks at the same data as the original
                        - No data is copied
                        - Changing one Affects the other
                    
                - [*] Deep copy: `.copy()`
                        - Creates a new array with its own data
                        - Any changes in the copy DOES NOT affect the original data

    -   [*] Pandas (ONE WEEK)
                # 📌 Pandas Checklist for Machine Learning
                '''
                Pandas is a Python library for data analysis and manipulation. It’s one of the most widely used tools in data science, finance, machine learning, and analytics.
                '''
                ## 🔹 Data Structures
                - [*] Create Series & DataFrames (from dicts, lists, arrays, CSV, Excel)
                        - `Series` --> One dimensional data
                        - `DataFrame` ---> 2-D labeled data
                        
                - [*] Inspect data: `.head()`, `.tail()`, `.info()`, `.describe()`, `.shape`, `.columns`, `.dtypes`
                        - df.head()
                            - Takes a parameter to return the first (x parameter) columns
                        
                        - df.tail()
                            - Returns the given parameter of last columns
                        
                        - df.describe()
                            - Gives a description of the dataset
                        
                        - df.column()
                            - Returns all columns in the dataset
                        
                        - df.dtypes() 
                            - Returns the data type of the given parameter
                        - df.info()
                            - Vivid description of the dataset
                        
                            

                ## 🔹 Indexing & Selection
                - [*] Select by label: `.loc`
                        - `.loc[]`
                            - It is used to select data by labels
                            - .loc[row,column]
                            - Use names and conditions
                            - To select a single row
                                - df.loc["a"]
                            - Multiple rows
                                - df.loc[["a","b"]]
                            - Select with conditions
                                - df.loc[df["Year of experience"] > 2]
                                
                - [*] Select by position: `.iloc`
                        - Intenger-based filtering
                        - Uses row/column numbers
                        - Select by single row
                            - df.iloc[1]
                        - Select multiple rows
                            - df.iloc[[1,2]
                        - Updating using iloc
                            - df.iloc[1,2] = 89
                            
                - [*] Conditional filtering (e.g., `df[df["col"] > 10]`)
                        - df[df["Name"]=="Maina"]
                        - df[df["Age"] >= 19]

                ## 🔹 Data Cleaning
                - [*] Handle missing values: `.isnull()`, `.dropna()`, `.fillna()`
                        - `.isnull()`
                            - Used to detecct missing values OR NaN
                            - It returns a DataFrame OR series of Boolean values
                            - df.isnull()
                        
                        - `.dropna()`
                            - Drop missing values
                        
                        - `.fillna()`
                            - Fill Missing values
                            - `df["Salary"] = df["Salary"].fillna(df["Salary"].mean())`
                            
                - [*] Handle duplicates: `.duplicated()`, `.drop_duplicates()`
                        -   `.duplicated()`
                                - Used to find duplicated values in a DataFrame / series
                                - df.duplicated(subset=None, keep="first")
                        -   `.drop_duplicated()`
                                - Drop all duplicated values in the given subset
                                - df.drop_duplicated()
                                
                - [*] Rename columns/index: `.rename()`
                        - Used to rename columns/rows
                        - df = df.rename(columns={"Dept": "Department"})
                        
                - [*] Change data types: `.astype()`
                        - Used to Convert the data type of a column or entire DataFrame
                        - df.astype(dtype, copy=True, errors='raise')

                ## 🔹 Transformation
                - [*] Apply functions: `.apply()`, `.map()`, `.applymap()`
                        - `.apply()`
                            - It is used to apply a function along an axis of a DataFrame(rows/Columns) or each element in series
                            - df.apply(Function, axis=0) # DataFrame
                            - series.apply(func) # for series
                            
                        - `.map()`
                            - Used to trnasform or map values in series != in DataFrame element wise
                            - series.map(arg)
                                - args can be:
                                    - Functions
                                    - Dictionaries keys
                                    - Other series
                                    
                        - `.applymap()`
                            - Used to apply function element-wise to every single value in a DataFrame != seriess
                            - df.applymap(func)
                                - func --> Function to apply to each element
                                
                - [*] String operations: `.str`
                        - It is an accessor that lets you apply string functions element-wise to a series(or a column) of strings (Optimzed for Vectors)
                        - Example functions
                            - .lower()  - .upper()   - .len()  - .contains()   - .replace()
                            - .split()  - .strip()   - .startswith()    - .endswith()   - .cat()
                            
                - [*] Replace values: `.replace()`
                        - Used to replace values in series or DataFrame
                        - `df.replace(to_replace, value=None, inplace=False, regex=False)`
                            - to_replace ---> The values to replace
                            - value ---> The New Value(s)
                            - regex ---> Use regex for pattern replacement
                            - inplace=True ---> Change data directly with creating a new object
                            
                ## 🔹 Grouping & Aggregation
                - [*] Group data: `.groupby()`
                        - Used to split data into groups,apply a function(like sum, mean, count, custom function) and the combine the results.
                        - `df.groupby("Department")["Salary"].mean()`

                    
                - [*] Aggregate: `.agg()`
                        - Used with series, DataFrame or after .groupby() to apply one or multiple aggregation functions. (summerize this data using one ot many funtions)
                        - `df.agg(func,axis=0)`
                        - `df.groupby("col").agg(func)`
                            - `axis=0` ---> Apply on columns
                            - `axis=`  ---> Apply on rows
                            
                - [*] Pivot tables: `.pivot_table()`
                        - Used to summerize and Aggregate data into a table format
                        - `pd.pivot_table(
                                data,\
                                values=None,\
                                index=None,\
                                columns=None,\
                                aggfunc="mean",\
                                fill_value=None,\
                                margins=False)`

                ## 🔹 Merging & Combining
                - [*] Merge DataFrames: `.merge()`
                        - Used to combine two DataFrames based on common columns or indices(just like SQL joints)
                        - pd.merge(
                             left,
                             right,
                             how="inner",
                             on=None,
                             left_on=None,
                             right_on=None,
                             left_index=False,
                             right_index=False,
                             suffixes=("_x","_y"))
                             
                        
                        - `left,right` ---> DataFrames to merge
                        - `how` ---> Type of join ("inner","outer","left","right")
                                - `inner` ---> Only matching rows
                                - `left` ---> all rows from left matching from right
                                - `right` ---> all rows from right matching from left
                                - `outer` ---> all rows from both(union) NaN if no matching
                        - `on` ---> Common columns to join on
                        - `left_on,right_on` ---> if the key column has different names in each DF
                        - `left_index,right_index` ---> Join on indexes
                        - `suffixes` ---> rename duplicate column names
                        - `df1.merge(df2,how="inner",sort=False,on="DeptID",suffixes=("_r","_l"))`
                        
                - [*] Concatenate: `.concat()`
                        - Used to combine multiple DataFrames OR series along rows or columns
                        - It doesn match keys it stacks/append
                        - pd.concat(
                                objs,
                                axis=0,
                                join="outer",
                                ignore_index=False,
                                keys=None)
                        - `objs` ---> List/tuple of DataFrames tor series
                        - `join` ---> How to handle mismatched columns
                        - `ignore_index` ---> True = Reset index after concatenation
                        - `keys` ---> Create hiearachial index to identify source
                        - combined_dfs = pd.concat([df1,df2],axis=0)
                        
                - [*] Join: `.join()`
                        - Used to join DataFrames - when joining on the index(row labels) instead of columns
                        - df1.join(
                              df2,
                              on=None,
                              how="left",
                              lsuffice="",
                              rsuffix="",
                              sort=False)
                        - `df2` ---> DataFrames to join
                        - `on` ---> Columns to join
                        - `how` ---> Type of join
                        - `lsuffix,rsuffix` ---> Add suffix if column names crash

                ## 🔹 Sorting & Ranking
                - [*] Sort by values: `.sort_values()`
                        - used to sort DataFrame or series by one or more column values(NOT INDEX)
                        - DataFrame.sort_values(
                                    by,
                                    axis=0,
                                    ascending=True,
                                    inplace=True,
                                    na_position="last",
                                    kind="quicksort",
                                    ignore_index=False)
                        - `df.sort_values("Salary",ascending=True,axis=0)`
                                    
                - [*] Sort by index: `.sort_index()`
                        - Used to sort a DataFrame or series by its index labels, instead of Column values
                        - DataFrame.sort_values(
                                    axis=0,
                                    ascending=True,
                                    inplace=True,
                                    kind="quicksort",
                                    na_position="last",
                                    ignore_index=False)
                        
                        - `df.sort_index(axis=0,ascending=True)`
                        
                        
                - [*] Ranking: `.rank()`
                        - Used to assign ranks to data with ties handled in different  ways
                        - df.rank(
                             axis=0,
                             method='average',
                             numeric_only=False,
                             na_option="keep",
                             ascending=True,
                             pact=False)
                        
                        - `methods` ---> How to assign ranks when there ties
                            - `average` ---> Average rank(default)
                            - `min` ---> lowest rank in group
                            - `max` ---> highest rank in group
                            - `first` ---> Assign ranks in irder they appear
                            - `dense` ---> like "min" but ranks increase by 1
                        - `na_option` ---> "keep", "top","bottom"
                        - `pct=True` ---> returns rank as percentage of data
                        - `df["Salary"].rank(axis=0,method="dense",ascending=True)`
                        
                ## 🔹 Input & Output
                - [*] Read/write CSV: `.read_csv()`, `.to_csv()`
                - [*] Read/write Excel: `.read_excel()`, `.to_excel()`
                - [*] Read/write JSON: `.read_json()`, `.to_json()`


    -   [ ] Matplotlib & Seaborn (visualization) (ONE WEEK)
                # 📌 Matplotlib Checklist for Machine Learning
                ## 🔹 Basic Plots
                - [*] Line plot: `plt.plot()`
                - [*] Scatter plot: `plt.scatter()`
                - [*] Histogram: `plt.hist()`
                - [*] Bar chart: `plt.bar()`

                ## 🔹 Plot Customization
                - [*] Add title: `plt.title()`
                - [*] Label axes: `plt.xlabel()`, `plt.ylabel()`
                - [*] Add legend: `plt.legend()`
                - [*] Set axis limits: `plt.xlim()`, `plt.ylim()`
                        
                        - plt.xlim()  Used to set limits of the x-axis
                        - plt.ylim()  Used to set limits of the y-axis
                        
                - [*] Add grid: `plt.grid()`

                ## 🔹 Figures & Axes
                - [*] Create figure: `plt.figure()`
                        - Used to create a new figure where you can plot(more control)
                - [*] Create subplots: `plt.subplot()`, `plt.subplots()`
                        - `plt.subplot()` ---> Used to create multiple plots inside a figure
                        - `plt.subplots()`---> More modern way to create a plots inside a figure
                        - `plt.subplots() - single plot`
                            - `plt.subplots(2,2,figsize=(6,6))`
                            
                - [*] Access Axes methods (`ax.plot`, `ax.scatter`, etc.)
                        - Both are methods for plotting data.
                        - `ax.plot` --> Draw lineplots but with optional markers; Best for continuous data.
                        - fig,ax = plt.sublots()
                            - ax.plot(x,y,color="blue",marker="o",linestyle="--",label="Line plot")
                        
                        - `ax.scatter()` ---> Draw scatter plots ONLY,no connecting dots.
                                - fig,ax = plt.sublots()
                                    ax.scatter(x,y,color="red",linestyle="+",marker="--",label="Scatter plot")
                        
                - [*] Adjust layout: `plt.tight_layout()`
                        - Used with `plt.subplots()` to automatically adjust subplot parameters so that title/labels do not overlap
                        
                ## 🔹 Styling
                - [*] Change colors & markers
                - [*] Line styles (`linestyle`, `linewidth`)
                - [*] Marker styles (`marker`, `markersize`)
                - [ ] Customize ticks & labels (`plt.xticks()`, `plt.yticks()`)

                ## 🔹 Saving Plots
                - [*] Save figure: `plt.savefig("plot.png")`

    -   📚 Resource: [Kaggle Python Course](https://www.kaggle.com/learn/python)
    
-   [*] Use Development Tools
    -   [*] Jupyter Notebook for experiments
    -   [*] NeoVim for projects
-   [*] Learn **Git & GitHub**
    -   [*] Git basics (init, add, commit, push)
    -   [*] Branching & merging
    -   [*] Upload projects to GitHub
    -   📚 Resource: [GitHub Skills](https://skills.github.com/)

📌 **Mini Project:** Titanic dataset → clean → visualize survival by gender/class.

------------------------------------------------------------------------

## 2. Data Handling

-   [*] Data Cleaning
    -   [*] Remove missing values
            - Remove entire rows with missing values
            - Remove entire colimns withs missing values
            - Remove data when;
                - The dataset is large and loosing some values is not a loss.
                - There are too many missing values > 70%.
                - Missing data is random.
                
            - When not to remove:
                - Missing data forms a pattern.
                - Dataset is small
                - When missing values carry meaning
            
    -   [*] Handle duplicates
            - A duplicate is when the same row appears more than once in a dataset.
            - `1` --> Inspect the duplicates.
            - `2` --> Remove them
            - `3` --> Keep the first or the first occurence
            
    -   [*] Correct inconsistent data
            - Inconsistent data it's when the information is recorded in different ways like `(nairobi, Nairobi)`
                - Test Case different
                - Spelling valiation/typo
                - Different formats i.e time
                - Units mismatch
                - Extra spaces and symbols
            
            - How to correct;
                - Stardadize case
                - Trim spaces and remove specail characters
                - Unify Formats
                - Stardadize categories
                - Fix typos
                - Domain knowledge rules --- `If Dept code = 101` always mean   `sales`
    
-   [*] Feature Encoding
    -   [*] Convert categorical → numeric (Label/One-Hot Encoding)
            - Transforming categorical(non-numeric)  data into numeric values
            - Must be implemented in a way that preserves meaning
            
            - `One-Hot Encoding`
                    - Create a new column of each category with 1/0 values
            - Binary enconding - First represent as integers then converted to binaries
            - Target/mean enconding - Replace each category with the mean of the target variable for the category
            - Frequency enconding - Replace each category with how man times it appears
            
    -   [*] Handle date/time features
            - Convert the category into datetime format.
            - `df["Date"].astype("datetime")`
            
-   [*] Scaling & Normalization
        - ` Scaling is the process of adjusting values of numeric features soo that they are on a similar range or distribution`
        - `Normalization is the process of rescaling data into a fixed range [0,1]`
        
    -   [*] StandardScaler (mean=0, var=1)
                - Transforms data soo it has mean=0  and stardard deviation = 1
                - Useful when data is normally distributed
                - ` z = (x - u) / a`
                
    -   [*] MinMaxScaler (range \[0,1\])
                - Resizes values into a fixed range of [0,1]
                - Useful when features must be bounded
                    
                           ` x-min(x)`
                    `x = --------------------`
                        `max(x) - min(x)`
                        
                        
-   [*] Work with Files & Databases
    -   [*] Load CSV, JSON, Excel
    -   [*] Query data with SQL
                - `INNER JOIN`  ---> Only matching rows
                - `LEFT JOIN` ---> all rows from left table, plus matches
                - `RIGHT JOIN` ---> All rows from right table,plus matches
                - `FULL JOIN` ---> All rows from both tables
                
                - `SELECT col FROM  table WHERE condition GROUP BY col HAVING condition ORDER BY col LIMIT n`
                    - SELECT ---> Choose columns
                    - WHERE ---> Filter rows
                    - GROUP BY + HAVING ---> Aggregate
                    - ORDER BY + LIMIT ---> Sort and restrict results
                    - JOIN ---> Combine multiple tables
                    
    -   [*] Use SQLite or PostgreSQL
    -   📚 Resource: [SQL for Data Science (CourseraFree)](https://www.coursera.org/learn/sql-for-data-science)

📌 **Mini Project:** Build a pipeline that cleans raw student records →
outputs usable dataset.

------------------------------------------------------------------------

## 3. Classical Machine Learning

-   [ ] Regression
    -   [*] Linear Regression
                # 📌 Linear Regression Checklist for Machine Learning
                    - Used to model relationship between independent variable(feature) and dependent variable(target) by fitting a straight line(Line of the best fit)
                ## 🔹 Theory
                - [*] Understand the concept of regression (predicting continuous values)
                        - Understand pattern in the data
                        - Predict outcomes (future or unknown)
                        - Measure strengths of relationships between variables
                        - Regression tries to fit a function( Line or a curve) that best explains the relationship between input(s) `X` and Output `Y`
                        
                - [*] Equation of linear regression: `y = β0 + β1*x + β2*x2 + βb3*x3 +....+βn*xn + ε`
                        - `y` ---> Target (dependent variable)
                        - `β0` ----> intercept (Value of y when x = 0)
                        - `β1,β2,β3...` -----> coefficients weights (Parameters)
                        - `x1, x2, x3` -----> Features(Independent varaibles)
                        - `ε` ---> Error term( Things model can't explain)
                        
                - [*] Role of coefficients (slope, intercept).
                        - `Intercept` (β0)
                            - The value of the dependent valiable `Y` when `x = 0`
                            - It sets the baseline for prediction
                        
                        - `Slopes`(`β1,β2,β3...`)
                            - The amount `Y` changes when `X` increases by one unit, keeping other variables constant
                            - Shows direction and strength of the relationship between predictors and the target
                                - Negative slope ---> `Y` decreases as `X` increases
                                - Positive slope ---> `Y` increases as `X` increases
                                
                - [*] Assumptions of linear regression:
                        - `Linearity`
                            - Relationship between `Y` and `X` is linear
                            - By fiiting a hyperplane in a non-linear relationship, the results will be inaccurate
                            
                        - `Independence of errors`
                            - The Residual(errors) `Y - p(Y)` should be independent of each other.
                            - Correlated erros(i.e In time series data) violate regression assumptions, leanding to underestimate stardard errors
                            
                        - `Homoscedasticity (constant variance)`
                            - The variance of the residuals should be constant across all levels of `X`
                            - If variance changes prediction of certain levels of X will less reliable
                            - Plot residuals verse fitted values should show a random spread not a funnel shape
                            
                        - `Normality of errors`
                            - The residuals should be approximately normally distributed
                            - Need for accurate confidence intervals and hypothesis tests.
                            - Histogram or Q-Q plot of residuals
                            
                - [*] Cost function:
                        - Cost function is the difference between the Actual value and the Predicted value.
                        - The smaller the cost, the better the model fits the data
                        - `Mean Squared Error (MSE)`
                            - Used for regression models (Prediction of numbers)
                            - It Calculates the average of the squared difference between predicted and actual values- Squaring penalizes larger errors more
                        
                        - `Mean Absolute Error(MAE)`
                            - Also for regression
                            - Measures the absolute difference between predicted and actual values
                            
                        - `Cross-Entropy Loss(log loss)`
                            - Used for classification problems, especially binary classification
                            - It penalizes prediction that are far from the true Label.
                            - Perfect prediction losss close to 0
                        
                        - Optimization algorithms like Gradient descent uses cost function to adjust model parameters
                        - Cost Function Evaluate model performance - Lower cost = Better model.
                        
                - [*] Optimization: Gradient Descent (intuition)
                        - Its an Optimization algorithms used to minimize cost funtions
                        - Like finding the lowes point in a valley;
                            - The valley - Cost function curve
                            - Height -  cost
                            - Goal - Reach the lowest point(minimum cost)
                        
                        - Adjust model parameters (weights `w` and biases `b` to ensure predicted values are as close as possible to the real values)
                            - Cost funtion tells `how bad` the current parameters are.
                            - Gradient descent tell us which direction to move to reduce the cost
                        
                        - `Components`
                            - `Learning Rate`
                                - Too small = Slow convergence
                                - Too large = overshoot minimum
                            
                            - `Gradient`
                                - Vector of partial derivatives the points towards steepest ascent
                                - We go in the opposite direction to minimize cost
                            
                            - `Iterations`
                                - Repeat updating parameters unitil cost stops decreasing significantly

                ## 🔹 Data Preparation (NumPy & Pandas)
                - [*] Load dataset (`pd.read_csv`, etc.)
                - [*] Inspect dataset (`.head()`, `.info()`, `.describe()`)
                - [*] Handle missing values (`.dropna()`, `.fillna()`)
                - [*] Feature selection (`df[["X"]]`, `df["y"]`)
                - [*] Split data into train/test sets (`train_test_split` from sklearn)

                ## 🔹 Visualization (Matplotlib & Seaborn)
                - [*] Scatter plot of features vs target (`plt.scatter`, `sns.scatterplot`)
                - [*] Regression line plot (`sns.regplot`, `sns.lmplot`)
                - [*] Residual plot to check assumptions
                - [*] Correlation heatmap (`sns.heatmap`)

                ## 🔹 Implementation
                - [*] Using NumPy:
                - Implement regression manually with normal equation
                - Predict new values
                - [*] Using scikit-learn:
                - Import model: `from sklearn.linear_model import LinearRegression`
                - Fit model: `.fit(X_train, y_train)`
                - Predict: `.predict(X_test)`
                - Get coefficients & intercept: `.coef_`, `.intercept_`

                ## 🔹 Model Evaluation
                - [*] Calculate metrics:
                - Mean Squared Error (MSE)
                - Root Mean Squared Error (RMSE)
                - Mean Absolute Error (MAE)
                - R² score (`r2_score`)
                - [*] Compare training vs test performance

                ## 🔹 Extensions
                - [*] Multiple linear regression (more than one feature)
                - [*] Polynomial regression (`PolynomialFeatures`)
                - [*] Regularization (Ridge, Lasso, Elastic Net)

    -   [ ] Logistic Regression
                # 📌 Logistic Regression Checklist for Machine Learning
                ## 🔹 Theory
                - [*] Understand classification vs regression
                - [*] Logistic regression equation:  
                `p(y=1|x) = 1 / (1 + e^-(β0 + β1*x))`
                - [*] Sigmoid function and interpretation of probabilities
                - [*] Decision boundary & threshold (default = 0.5)
                - [*] Assumptions of logistic regression:
                - No extreme multicollinearity
                - Linear relationship between log-odds and predictors
                - Large sample size preferred
                - [*] Cost function: Log Loss (Cross-Entropy)
                - [*] Optimization: Gradient Descent (intuition)

                ## 🔹 Data Preparation (NumPy & Pandas)
                - [*] Load dataset (`pd.read_csv`, etc.)
                - [*] Inspect dataset (`.head()`, `.info()`, `.describe()`)
                - [*] Handle missing values (`.dropna()`, `.fillna()`)
                - [*] Encode categorical variables (`pd.get_dummies`, `LabelEncoder`)
                - [*] Feature scaling (`StandardScaler`, `MinMaxScaler`)
                - [*] Split data into train/test sets (`train_test_split`)

                ## 🔹 Visualization (Matplotlib & Seaborn)
                - [*] Explore target distribution (`sns.countplot`)
                - [*] Feature vs target relationship (`sns.boxplot`, `sns.violinplot`)
                - [*] Correlation heatmap (`sns.heatmap`)
                - [*] Decision boundary visualization (2D datasets)

                ## 🔹 Implementation
                - [*] Using scikit-learn:
                        - [*] Import: `from sklearn.linear_model import LogisticRegression`
                        - [*] Fit model: `.fit(X_train, y_train)`
                        - [*] Predict class labels: `.predict(X_test)`
                        - [*] Predict probabilities: `.predict_proba(X_test)`
                        - [*] Get coefficients & intercept: `.coef_`, `.intercept_`

                ## 🔹 Model Evaluation
                - [*] Accuracy score
                - [*] Confusion matrix (`confusion_matrix`)
                - [*] Classification report (`precision`, `recall`, `f1-score`)
                - [*] ROC curve & AUC (`roc_curve`, `auc`)
                - [*] Precision-Recall curve
                - [*] Cross-validation performance

                ## 🔹 Extensions
                - [ ] Multiclass classification (One-vs-Rest, One-vs-One)
                - [ ] Regularization (L1 = Lasso, L2 = Ridge, Elastic Net)
                - [ ] Hyperparameter tuning (`C`, `penalty`, `solver`)

-   [ ] Classification
    -   [ ] Decision Trees
    -   [ ] Random Forests
    -   [ ] k-NN
    -   [ ] Naive Bayes
    -   [ ] Support Vector Machines (SVM)
-   [ ] Clustering
    -   [ ] k-Means
    -   [ ] DBSCAN
    -   [ ] PCA (dimensionality reduction)
-   [ ] Model Evaluation
    -   [ ] Accuracy
    -   [ ] Precision, Recall, F1
    -   [ ] Confusion Matrix
    -   [ ] ROC & AUC
-   📚 Resource: [Scikit-learn UserGuide](https://scikit-learn.org/stable/user_guide.html)

📌 **Projects:**\
- House price prediction (regression)\
- Spam email classifier (classification)\
- Customer segmentation (clustering)

------------------------------------------------------------------------

## 4. Deep Learning

-   [ ] Basics of Neural Networks
    -   [ ] Layers & neurons
    -   [ ] Activation functions
    -   [ ] Backpropagation (concept)
-   [ ] CNNs (Computer Vision)
    -   [ ] Convolutions & pooling
    -   [ ] Image classification
-   [ ] RNNs & LSTMs (Text & Sequences)
    -   [ ] Sequence prediction
    -   [ ] Sentiment analysis
-   [ ] Transformers (Modern NLP)
    -   [ ] Attention mechanism
    -   [ ] Pretrained models (BERT, GPT)
-   [ ] Frameworks
    -   [ ] TensorFlow
-   📚 Resource: [DeepLearning.AI TensorFlow Specialization](https://www.coursera.org/specializations/tensorflow-in-practice)

📌 **Projects:**\
- MNIST digit classifier\
- IMDb sentiment analysis

------------------------------------------------------------------------

## 5. Advanced ML

-   [ ] Feature Engineering
    -   [ ] Create custom features
    -   [ ] Feature selection
-   [ ] Ensemble Methods
    -   [ ] Gradient Boosting
    -   [ ] XGBoost
    -   [ ] LightGBM
    -   [ ] CatBoost
-   [ ] Hyperparameter Tuning
    -   [ ] GridSearchCV
    -   [ ] RandomizedSearchCV
    -   [ ] Optuna
-   [ ] Explainability
    -   [ ] LIME
    -   [ ] SHAP
-   📚 Resource: [Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow(Book)](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/)

📌 **Projects:**\
- Loan approval prediction\
- Fraud detection system\
- Kaggle beginner competition

------------------------------------------------------------------------

## 6. ML Deployment & MLOps

-   [ ] APIs
    -   [ ] Build ML API with Flask
    -   [ ] Build ML API with FastAPI
-   [ ] Containers
    -   [ ] Learn Docker basics
    -   [ ] Containerize ML models
-   [ ] CI/CD for ML
    -   [ ] Automate training pipeline
    -   [ ] Continuous deployment
-   [ ] Cloud ML
    -   [ ] AWS SageMaker
    -   [ ] GCP Vertex AI
    -   [ ] Azure ML
-   [ ] Monitoring & Retraining
    -   [ ] Track model drift
    -   [ ] Auto retrain pipeline
-   📚 Resource: [Made With ML (MLOps Guide)](https://madewithml.com/)

📌 **Projects:**\
- Deploy a movie recommendation API\
- Real-time object detection app (Flask + webcam)

------------------------------------------------------------------------

## 7. Specialization (Pick One)

-   [ ] **Computer Vision**
    -   [ ] Image classification
    -   [ ] Object detection
    -   [ ] Face recognition
-   [ ] **NLP**
    -   [ ] Chatbots
    -   [ ] Text summarization
    -   [ ] Translation
-   [ ] **Cybersecurity + ML**
    -   [ ] Intrusion detection
    -   [ ] Malware detection
-   [ ] **Finance + ML**
    -   [ ] Fraud detection
    -   [ ] Credit scoring
    -   [ ] Risk prediction
-   📚 Resource: [Fast.ai Specializations](https://course.fast.ai/)

📌 **Capstone Project:** End-to-end ML system → collect data →
preprocess → train → deploy → visualize.

------------------------------------------------------------------------

# 🎯 Success Checklist

-   [ ] Build portfolio projects on GitHub\
-   [ ] Join Kaggle & solve competitions\
-   [ ] Share projects on LinkedIn\
-   [ ] Keep learning from ML papers/blogs\
-   [ ] Stay updated with frameworks & tools

