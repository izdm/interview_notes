# 回溯算法

## 77.组合问题

    func combine(n int, k int) [][]int {
        res := make([][]int, 0)
        path := make([]int, 0,k)
        var travelsal func(n int, k int, startIndex int)
        travelsal = func(n int, k int, startIndex int) {
            if len(path) == k {
                tmp :=make([]int, len(path))
                copy(tmp,path)
                res = append(res, tmp)
                return
            }
            for i := startIndex; i <= n; i++ {
                path = append(path, i)
                travelsal(n, k, i+1)
                path = path[:len(path)-1]
            }
        }
        travelsal(n, k,  1)
        return res
    }

经典回溯

## 216.组合总和III

    func combinationSum3(n int, k int) [][]int {
        res := make([][]int, 0)
        path := make([]int, 0, k)
    
        // 修改为传递当前的 sum
        var travelsal func(n int, k int, startIndex int, sum int)
        travelsal = func(n int, k int, startIndex int, sum int) {
            // 当路径长度等于k且和等于n时，记录结果
            if len(path) == k {
                if sum == n {
                    tmp := make([]int, len(path))
                    copy(tmp, path)
                    res = append(res, tmp)
                }
                return
            }
    
            for i := startIndex; i <= 9; i++ {
                path = append(path, i)      // 选择数字
                travelsal(n, k, i+1, sum+i) // 递归时传递新的sum
                path = path[:len(path)-1]   // 回溯：撤销选择，回到上一层
            }
        }
    
        travelsal(n, k, 1, 0) // 初始时，sum从0开始
        return res
    }

## 17.电话号码的字母组合

    func letterCombinations(digits string) []string {
    
        info :=[]string{"","","abc","def","ghi","jkl","mno","pqrs","tuv","wxyz"}
        var res []string
        var path []byte
        if digits ==""{
            return res
        }
        var backtracking func(digits string,index int)
        backtracking =func(digits string,index int){
            if(len(path)==len(digits)){
                tmp :=string(path)
                res = append(res,tmp)
                return
            }
            digit := int(digits[index]-'0')
            for i:=0;i<len(info[digit]);i++{
                path = append(path,info[digit][i])
                backtracking(digits,index+1)
                path = path[:len(path)-1]
            }
        }
        backtracking(digits,0)
        return res
    }

## 39.组合总数

    func combinationSum(candidates []int, target int) [][]int {
        var res [][]int
        var path []int
        sum :=0
        var backtracking func(candidates []int,target int,sum int)
        backtracking = func(candidates []int,target int,sum int){
            if(sum >target){
                return
            }
            if(sum == target){
                //一定要这么写 ，不然会报错
                tmp := make([]int, len(path))
                copy(tmp, path)
                res = append(res, tmp)
                return
            }
            for i:=0;i<len(candidates);i++{
                path = append(path,candidates[i])
                backtracking(candidates,target,sum+candidates[i])
                path = path[:len(path)-1]
            }
        }
        backtracking(candidates,target,sum)
        return res
    }

错误写法 这样子输出的是排列 不是组合 ，一样要控制一个index
    func combinationSum(candidates []int, target int) [][]int {
        var res [][]int
        var path []int
        sum :=0
        var backtracking func(candidates []int,target int,index int ,sum int)
        backtracking = func(candidates []int,target int,index int ,sum int){
            if(sum >target){
                return
            }
            if(sum == target){
                //一定要这么写 ，不然会报错
                tmp := make([]int, len(path))
                copy(tmp, path)
                res = append(res, tmp)
                return
            }
            for i:=index;i<len(candidates);i++{
                path = append(path,candidates[i])
                backtracking(candidates,target,i,sum+candidates[i])
                path = path[:len(path)-1]
            }
        }
        backtracking(candidates,target,0,sum)
        return res
    }

## 40 组合综合II

    func combinationSum2(candidates []int, target int) [][]int {
        var res [][]int
        var path []int 
        sort.Ints(candidates)
        var dfs func(candidates []int,target int,start int,sum int)
        dfs = func(candidates []int,target int,start int,sum int){
            if sum>target{
                return
            }
            if sum ==target{
                tmp :=make([]int,len(path))
                copy(tmp,path)
                res = append(res,tmp)
                return
            }
    
            for i:=start;i<len(candidates);i++{
                path=append(path,candidates[i])
                dfs(candidates,target,i+1,sum+candidates[i])
                path=path[:len(path)-1]
            }
        }
        dfs(candidates,target,0,0)
        return res
    }

错误写法，没有考虑到相同的组合重复出现的情况，因为元素是有重复的
    func combinationSum2(candidates []int, target int) [][]int {
        var res [][]int
        var path []int

        // 排序以便能够跳过重复的元素
        sort.Ints(candidates)

        var dfs func(candidates []int, target int, start int, sum int)
        dfs = func(candidates []int, target int, start int, sum int) {
            // 如果和大于目标，剪枝，停止递归
            if sum > target {
                return
            }

            // 如果和等于目标，记录结果
            if sum == target {
                tmp := make([]int, len(path))
                copy(tmp, path)
                res = append(res, tmp)
                return
            }

            for i := start; i < len(candidates); i++ {
                // 跳过重复元素
                if i > start && candidates[i] == candidates[i-1] {
                    continue
                }

                path = append(path, candidates[i])               // 选择当前数字
                dfs(candidates, target, i+1, sum+candidates[i])   // 递归时从下一个位置开始
                path = path[:len(path)-1]                         // 回溯
            }
        }

        dfs(candidates, target, 0, 0)  // 从索引0开始
        return res
    }

先排序，如果在递归时，如果当前元素和前一个元素相同，并且前一个元素在当前递归层已经被跳过了，就跳过当前元素，避免重复组合。

## 78.子集

    func subsets(nums []int) [][]int {
        var res [][]int
         path :=[]int{}
        res = append(res,path)
        var backtracking func(nums []int,start int)
        backtracking =func(nums []int,start int){
            if(start==len(nums)){
                return
            }
            for i:=start;i<len(nums);i++{
                path =append(path,nums[i])
                tmp :=make([]int,len(path))
                copy(tmp,path)
                res = append(res,tmp)
                backtracking(nums,i+1)
                path =path[:len(path)-1]
            }
        }
        backtracking(nums,0)
        return res
    }

先加入空集，后续只要对数组有改动就添加进去

# 二叉树

## 144.二叉树前序遍历

    方法二函数内部又定义了一个函数
        /**
         * Definition for a binary tree node.
         * type TreeNode struct {
         *     Val int
         *     Left *TreeNode
         *     Right *TreeNode
         * }
         */
        //写法一
        func preorderTraversal(root *TreeNode) []int {
            var res []int
            preorder(root, &res)
            return res
        }
        func preorder(root *TreeNode, res *[]int) {
            if root == nil {
                return
            }
            *res = append(*res, root.Val)
            preorder(root.Left, res)
            preorder(root.Right, res)
        }
    
    
        //写法二
        func preorderTraversal(root *TreeNode) []int {
            res := []int{}
            //先声明
            var traversal func(root *TreeNode)
            traversal = func(root *TreeNode) {
                if root == nil {
                    return
                }
                res = append(res, root.Val)
                traversal(root.Left)
                traversal(root.Right)
            }
            traversal(root)
            return res
        }

## 226.翻转二叉树

    /**
     * Definition for a binary tree node.
     * type TreeNode struct {
     *     Val int
     *     Left *TreeNode
     *     Right *TreeNode
     * }
     */
    func invertTree(root *TreeNode) *TreeNode {
        if root == nil {
            return root
        }
    
        var swap func(root *TreeNode)
        swap = func(root *TreeNode) {
            if root == nil {
                return
            }
            root.Left,root.Right = root.Right,root.Left
            swap(root.Left)
            swap(root.Right)
        }
        swap(root)
        return root
    }

## 101.对称二叉树

    根节点两边分别在内侧和外侧比较
        /**
         * Definition for a binary tree node.
         * type TreeNode struct {
         *     Val int
         *     Left *TreeNode
         *     Right *TreeNode
         * }
         */
        func isSymmetric(root *TreeNode) bool {
            if root == nil {
                return true
            }
    
            var swap func(left *TreeNode,right *TreeNode) bool
            swap = func(left *TreeNode,right *TreeNode) bool{
                if(left == nil&&right==nil){
                    return true
                }
                if(left == nil||right==nil){
                    return false
                }
                if(left.Val!=right.Val){
                    return false
                }
    
                return swap(left.Left,right.Right)&&swap(right.Left,left.Right)
            }
            res :=swap(root.Left,root.Right)
            return res
        }
    
    ## 

## 104.二叉树的最大深度

### 写法1

    每向下一层，最大深度就增加1，看谁的深度最大
        func maxDepth(root *TreeNode) int {
            if root == nil {
                return 0
            }
            depth := 0
            var compare func(root *TreeNode, deptht int)
            compare = func(root *TreeNode, deptht int) {
                deptht = deptht + 1
                depth = max(deptht, depth)
                if root.Left != nil {
                    compare(root.Left, deptht)
                }
                if root.Right != nil {
                    compare(root.Right, deptht)
                }
    
            }
            compare(root, 0)
            return depth
        }
        func max(a, b int) int {
            if a > b {
                return a
            }
            return b
        }

### 写法2

    func maxDepth(root *TreeNode) int {
        if root==nil{
            return 0
        }
        return max(maxDepth(root.Left)+1,maxDepth(root.Right)+1)
    }
    func max(a, b int) int {
        if a > b {
            return a
        }
        return b
    }

## 111.二叉树的最小深度

    每一个节点都返回下面最小的两克树到根节点的最小值
        /**
         * Definition for a binary tree node.
         * type TreeNode struct {
         *     Val int
         *     Left *TreeNode
         *     Right *TreeNode
         * }
         */
        func minDepth(root *TreeNode) int {
            if root == nil {
                return 0
            }
    
            var traversal func(root *TreeNode) int
            traversal = func(root *TreeNode) int {
                if root.Left == nil && root.Right == nil {
                    return 1
                }
                leftnums:=0
                rightnums:=0
                if root.Left != nil {
                    leftnums=  traversal(root.Left)
                }
                if root.Right != nil {
                    rightnums= traversal(root.Right)
                }
                if root.Left==nil{
                    return rightnums+1
                }
                if root.Right==nil{
                    return leftnums+1
                }
                return min(leftnums+1, rightnums+1)
            }
            res := traversal(root)
            return res
        }
        func min(a, b int) int {
            if a < b {
                return a
            }
            return b
        }

## 222.完全二叉树的节点个数（easy）

    func countNodes(root *TreeNode) int {
        count:=0
        var traversal func(root *TreeNode)
        traversal = func(root *TreeNode){
            if(root==nil){
                return
            }
            count++
            traversal(root.Left)
            traversal(root.Right)
        }
        traversal(root)
        return count
    }

## 110.平衡二叉树

    我们用-1来当作flag，记录每一个节点是否是平衡节点，如果节点不平衡，则把flag写做-1否则就记录这棵树的高度
        func isBalanced(root *TreeNode) bool {
            if root==nil{
                return true
            }
    
            var travelsal func(root *TreeNode) int
            travelsal =func(root *TreeNode) int{
                if(root==nil){
                    return 0
                }
                l,r := travelsal(root.Left),travelsal(root.Right)
                if l==-1||r==-1{
                    return-1
                }
                if(l-r>1||r-l>1){
                    return -1
                }
                return max(l,r)+1
            }
            res :=travelsal(root)
            if res==-1{
                return false
            }
            return true
        }
    
        func max(a, b int) int {
            if a > b {
                return a
            }
            return b
        }

## 257.二叉树的所有路径

不断地更新路径，在根节点再推送
    /**
     * Definition for a binary tree node.
     * type TreeNode struct {
     *     Val int
     *     Left *TreeNode
     *     Right *TreeNode
     * }
     */
    func binaryTreePaths(root *TreeNode) []string {
        res := make([]string, 0)
        if root.Left == nil && root.Right == nil {
            res = append(res, strconv.Itoa(root.Val))
            return res
        }
        var travelsal func(root *TreeNode, tmp string)
        travelsal = func(root *TreeNode, tmp string) {
            if root == nil {
                return
            }
            if root.Left == nil && root.Right == nil {
                tmp += strconv.Itoa(root.Val)
                res = append(res, tmp)
                return
            }
            travelsal(root.Left, tmp+strconv.Itoa(root.Val)+"->")
            travelsal(root.Right, tmp+strconv.Itoa(root.Val)+"->")
        }
        travelsal(root,"")
        return res
    }

## 404.左叶子之和

正常遍历 是左叶子则加val
    func sumOfLeftLeaves(root *TreeNode) int {
        res := 0
        if root == nil {
            return res
        }
        var traversal func(root *TreeNode)
        traversal = func(root *TreeNode) {
            if root == nil {
                return
            }

            if root.Left != nil &&root.Left.Left==nil&&root.Left.Right==nil{//判断是不是左叶子
                res += root.Left.Val
            }
            traversal(root.Left)
            traversal(root.Right)
        }
        traversal(root)
        return res
    }

方法二递归法
    func sumOfLeftLeaves(root *TreeNode) int {
        if root == nil {
            return 0
        }
        leftValue := sumOfLeftLeaves(root.Left)   // 左

        if root.Left != nil && root.Left.Left == nil && root.Left.Right == nil {
            leftValue = root.Left.Val             // 中
        }

        rightValue := sumOfLeftLeaves(root.Right) // 右

        return leftValue + rightValue
    }

## 513.找树左下角的值

中序遍历，如果deep的值最大则保存该节点
    func findBottomLeftValue(root *TreeNode) int {
        res :=0
        maxDeep:=0

        var preorder func(root *TreeNode,deep int)
        preorder =func(root *TreeNode,deep int){
            if root==nil{
                return
            }

            preorder(root.Left,deep+1)
            if(deep>maxDeep){
                maxDeep =deep
                res = root.Val
            }
            preorder(root.Right,deep+1)
        }
        preorder(root,1)
        return res
    }

## 112.路径总和

不断地向下传路径
    func hasPathSum(root *TreeNode, targetSum int) bool {
        res :=false
        if(root==nil){
            return false
        }
        var travelsal func(root *TreeNode,tmp int)
        travelsal =func(root *TreeNode,tmp int){
            if(root==nil){
                return 
            }
            if(tmp+root.Val==targetSum&&root.Left==nil&&root.Right==nil){
                res = true
                return 
            }
            travelsal(root.Left,tmp+root.Val)
            travelsal(root.Right,tmp+root.Val)
        }
        travelsal(root,0)
        return res
    }

# 贪心算法

## 455.发放饼干

    func findContentChildren(g []int, s []int) int {
        if(len(s)==0) {
        return 0
        }
    
        sort.Ints(g)
        sort.Ints(s)
    
        count := 0          //合计总数
        index := len(s) - 1 //饼干下标
        for i := len(g) - 1; i >= 0; i-- {
            for index >= 0 && s[index] >= g[i] {
                count++
                index--
                break
            }
        }
        return count
    }

一旦看到数组 一定要确保边界问题

遍历胃口数组，与饼干数组匹配

## 376.摆动序列

默认最后一个元素为一个摆动，往前面加一个predif
    func wiggleMaxLength(nums []int) int {
        // 如果数组长度小于等于 1，直接返回数组长度
        if len(nums) <= 1 {
            return len(nums)
        }

        // 初始化前后差值
        previousDifference := 0 // 前一个差值
        currentDifference := 0  // 当前差值
        // 摆动序列的长度，初始值为 1（至少包含一个元素）
        count := 1 

        // 遍历数组，比较相邻元素差值
        for i := 0; i < len(nums)-1; i++ {
            currentDifference = nums[i+1] - nums[i]
            // 判断是否形成摆动
            if (previousDifference >= 0 && currentDifference < 0) || 
               (previousDifference <= 0 && currentDifference > 0) {
                count++                     // 摆动计数加 1
                previousDifference = currentDifference // 更新前一个差值
            }
        }

        // 返回最大摆动子序列的长度
        return count
    }

## 53.最大子数组之和

就看每一个数组元素的前面的pre是否大于0，大于0则加，小于零则是副总用直接舍弃

**贪心的思路为局部最优：当前“连续和”为负数的时候立刻放弃，从下一个元素重新计算“连续和”，因为负数加上下一个元素 “连续和”只会越来越小。从而推出全局最优：选取最大“连续和”**
    func maxSubArray(nums []int) int {
        nums2 := make([]int, len(nums))
        pre := 0 //
        cur := 0
        for i := 0; i < len(nums); i++ {
            if pre > 0 {
                cur = pre + nums[i]
            } else {
                cur = nums[i]
            }
            nums2[i] = cur
            pre = cur
        }
        result := nums2[0]
        for i := 0; i < len(nums2); i++ {
            if nums2[i] > result {
                result = nums2[i]
            }
        }
        return result
    }

这是我的代码
    func maxSubArray(nums []int) int {
        // 定义两个变量用于动态规划
        maxSum := nums[0]    // 最大子数组和，初始为数组第一个元素
        currentSum := nums[0] // 当前子数组和，初始为数组第一个元素

        // 遍历数组，从第二个元素开始计算
        for i := 1; i < len(nums); i++ {
            // 如果当前子数组和加上当前元素还不如当前元素本身大，则重置子数组
            if currentSum > 0 {
                currentSum += nums[i]
            } else {
                currentSum = nums[i]
            }

            // 更新最大子数组和
            if currentSum > maxSum {
                maxSum = currentSum
            }
        }

        return maxSum
    }

GPT优化代码

## 122.买卖股票的最佳时机

    func maxProfit(prices []int) int {
        if len(prices) <= 1 {
            return 0
        }
        result := 0
        for i := 1; i < len(prices); i++ {
            if prices[i]-prices[i-1] > 0 {
                result += prices[i] - prices[i-1]
            }
        }
        return result
    }

只要有上涨的地方 我就收入囊中

## 55.跳跃游戏

不断的调整覆盖范围
    func canJump(nums []int) bool {
        cover := 0
        for i := 0; i <= cover; i++ {
            cover = max(cover, i+nums[i])
            if cover >= len(nums)-1 {
                return true
            }
        }
        return false

    }

## 45.跳跃游戏II（未明白）

## 1005.K次取反后最大化的数组

抓住要点 k的值和负数谁大？k大则数组可以先全部变为正数，(k-nagetive)%2如果等于0则就是结果 否则的话就把最小的正数变成正数。如果k小 则把最小的负数变成正数
    func largestSumAfterKNegations(nums []int, k int) int {
        var result int
        sort.Ints(nums)
        negativenums := 0
        zeroflag := false
        for i := 0; i < len(nums); i++ {
            if nums[i] < 0 {
                negativenums++
            }
            if nums[i] == 0 {
                zeroflag = true
            }
        }
        if k <= negativenums {
            for i := 0; i < k; i++ {
                nums[i] = -nums[i]
            }
            sort.Ints(nums)
            for i := 0; i < len(nums); i++ {
                result = result + nums[i]
            }
            return result
        } else {
            for i := 0; i < negativenums; i++ {
                nums[i] = -nums[i]
            }
            sort.Ints(nums)
            if zeroflag {
                for i := 0; i < len(nums); i++ {
                    result = result + nums[i]
                }
                return result
            } else {
                if (k-negativenums)%2 == 1 {
                    nums[0] = -nums[0]
                }
                for i := 0; i < len(nums); i++ {
                    result = result + nums[i]
                }
                return result
            }
        }
    }

GPT优化代码
    func largestSumAfterKNegations(nums []int, k int) int {
        // 对数组进行升序排序
        sort.Ints(nums)

        // 处理负数，将负数变为正数
        for i := 0; i < len(nums) && k > 0 && nums[i] < 0; i++ {
            nums[i] = -nums[i]
            k--
        }

        // 再次排序，确保最小的数在前
        sort.Ints(nums)

        // 如果 k 仍为奇数，则将最小的数取反
        if k%2 == 1 {
            nums[0] = -nums[0]
        }

        // 计算最终数组的总和
        result := 0
        for _, num := range nums {
            result += num
        }

        return result
    }

## 134.加油站

贪心算法
    func canCompleteCircuit(gas []int, cost []int) int {
        cursum := 0
        totalsum := 0 //用来确保总的sum是大于等于0的 其实可以遍历一遍数组来确保是否大于0
        start := 0
        for i := 0; i < len(gas); i++ {
            cursum += gas[i] - cost[i]
            totalsum += gas[i] - cost[i]
            if cursum < 0 { //如果前面累加小于0，那么从下一个标点开始
                start = i + 1
                cursum = 0
            }
        }
        if totalsum < 0 {
            return -1
        }
        return start
    }

## 135.分发糖果(难)

    func candy(ratings []int) int {
        count := 0
        nums := make([]int, len(ratings))
        for i := 0; i < len(ratings); i++ {
            nums[i] = 1
        }
        for i := 0; i < len(ratings)-1; i++ {
            if ratings[i+1]-ratings[i] > 0 {
                nums[i+1] = nums[i] + 1
            }
        }
        for i := len(ratings) - 1; i > 0; i-- {
            if ratings[i-1] > ratings[i] {
                nums[i-1] = max(nums[i-1], nums[i]+1)
            }
        }
        for i := 0; i < len(ratings); i++ {
            count += nums[i]
        }
        return count
    }
    func max(a, b int) int {
        if a > b {
            return a
        } else {
            return b
    
        }
    }

从左往右边，确保如果i+1比i大，则nums[i+1]=nums[i]+1

然后再从右边向左，看

## 860.柠檬水找零

    func lemonadeChange(bills []int) bool {
        money := map[int]int{}
        for _, bill := range bills {
            if bill == 5 {
                money[5]++
            } else if bill == 10 {
                if money[5] >= 1 {
                    money[10]++
                    money[5]--
                } else {
                    return false
                }
            } else {
                if money[5] >= 1 && money[10] >= 1 {
                    money[5]--
                    money[10]--
                } else if money[5] >= 3 {
                    money[5] = money[5] - 3
                } else {
                    return false
                }
            }
        }
        return true
    }

如果是5块 就收进来，如果是10 ，20看看有没有找的，然后更新map

## 452.用最少量的箭引爆气球

        func findMinArrowShots(points [][]int) int {
            res := 1
            //先按左边的边界排序
            sort.Slice(points, func (i,j int) bool {
                return points[i][0] < points[j][0]
            })
            //
            for i := 1; i < len(points); i++ {
                //如果前一个右边界小于后一个左边界，直接++
                if points[i-1][1] < points[i][0] {
                    res++
                } else {
                    //非则说明有重叠部分，两个重叠部分按最小的右边算
                    points[i][1] = min(points[i-1][1], points[i][1])
                }
            }
    
            return res
        }



## 435.无重叠部分

    func eraseOverlapIntervals(intervals [][]int) int {
        res :=0
        //按照左端排序
        sort.Slice(intervals,func(i,j int)bool{
            return intervals[i][0]<intervals[j][0]
        })
    
        for i:=1;i<len(intervals);i++{
            //无重叠部分，res不用管
            if intervals[i][0]>=intervals[i-1][1]{
                continue
                //重叠 且右端比上一个短，相当于舍弃上一个，res直接++
            }else if intervals [i][0]<intervals[i-1][1]&&intervals[i][1]<=intervals[i-1][1]{
                res++
                continue
                //重叠且右端比上一个右端更长，直接删除此段，res++
            }else{
                intervals[i][0]=intervals[i-1][0]
                intervals[i][1]=intervals[i-1][1]
                res++
            }
        }
        return res
    }

## 763.划分字母区间

    func partitionLabels(s string) []int {
        var data [][]int
        var res []int
        //需要把每一个出现的字母的第一次和最后一次加入到数组中  与前两题一样的思路，和前面的区间比较
        for i := 0; i < len(s); i++ {
            for j := len(s) - 1; j >= i; j-- {
                if s[i] == s[j] {
                    data = append(data, []int{i, j})
                    break
                }
            }
        }
        //排序
        sort.Slice(data, func(i, j int) bool {
            return data[i][0] < data[j][0]
        })
    
        if len(data) == 1 {
            res = append(res, len(s))
            return res
        }
    
        for i := 1; i < len(data); i++ {
            if data[i][0] > data[i-1][1] {
                res = append(res, data[i-1][1]-data[i-1][0]+1)
            } else if data[i][0] < data[i-1][1] && data[i][1] < data[i-1][1] {
                data[i][0] = data[i-1][0]
                data[i][1] = data[i-1][1]
            } else {
                data[i][0] = data[i-1][0]
            }
            //结束的最后一次要加进去
            if i == len(data)-1 {
                res = append(res, data[i][1]-data[i][0]+1)
            }
        }
    
        return res
    }

## 56.合并区间

    func merge(intervals [][]int) [][]int {
        var res [][]int
        sort.Slice(intervals, func(i, j int) bool {
            return intervals[i][0] < intervals[j][0]
        })
    
        if len(intervals) == 1 {
            res = append(res, intervals[0])
            return res
        }
        for i := 1; i < len(intervals); i++ {
            if intervals[i][0] > intervals[i-1][1] {
                res = append(res, intervals[i-1])
            } else if intervals[i][0] <= intervals[i-1][1] && intervals[i][1] <= intervals[i-1][1] {
                intervals[i][0] = intervals[i-1][0]
                intervals[i][1] = intervals[i-1][1]
            } else if intervals[i][0] <= intervals[i-1][1] && intervals[i][1] > intervals[i-1][1] {
                intervals[i][0] = intervals[i-1][0]
            }
            if i == len(intervals)-1 {
                res = append(res, intervals[i])
            }
        }
    
        return res
    
    }

# 动态规划

五部曲

创建dp数组，递推公式，dp数组如何初始化，遍历顺序，打印dp数组（debug）

## 70.爬楼梯

    func climbStairs(n int) int {
        dp :=make([]int,46)
        dp[1]=1
        dp[2]=2
        for i:=3;i<=45;i++{
            dp[i]=dp[i-1]+dp[i-2]
        }
        return dp[n]
    }

## 746.使用最小的花费爬楼梯

    func minCostClimbingStairs(cost []int) int {
        dp :=make([]int,len(cost)+1)
        dp[0]=0
        dp[1]=0
        for i:=2;i<=len(cost);i++{
            dp[i]=min(cost[i-2]+dp[i-2],cost[i-1]+dp[i-1])
        }
        return dp[len(cost)]
    }
    func min(a,b int) int{
        if a<b{
            return a
        }else{
            return b
        }
    }

dp为爬到当前楼梯花费的力气

## 62.不同路径

dp数组存放的是到达此地的方式，因为题目只能向下向右走，所以只需要加上上面和左边的就行
    func uniquePaths(m int, n int) int {
        //dp数组的含义是，到达该地的路径数量，因为题目要求只能向右和向下走
        dp := [100][100]int{}
        for i := 0; i < 100; i++ {
            dp[0][i] = 1
        }
        for i := 0; i < 100; i++ {
            dp[i][0] = 1
        }
        for i := 1; i < 100; i++ {
            for j := 1; j < 100; j++ {
                dp[i][j] = dp[i][j-1] + dp[i-1][j]
            }
        }

        return dp[m-1][n-1]
    }

## 63.不同路径II

    func uniquePathsWithObstacles(obstacleGrid [][]int) int {
        lenght := len(obstacleGrid[0])
        width := len(obstacleGrid)
    
        // 先创建一个包含width个元素的一维数组切片，每个元素都是int类型的切片（此时内层切片还未初始化）
        dp := make([][]int, width)
        for i := range dp {
            // 为每个外层切片元素初始化一个长度为lenght的内层一维数组切片
            dp[i] = make([]int, lenght)
        }
    
        // 初始化第一行
        for i := 0; i < lenght; i++ {
            if obstacleGrid[0][i]!= 1 {
                dp[0][i] = 1
            } else {
                // 如果遇到障碍物，后面的路径数都为0
                break
            }
        }
    
        // 初始化第一列
        for i := 0; i < width; i++ {
            if obstacleGrid[i][0]!= 1 {
                dp[i][0] = 1
            } else {
                // 如果遇到障碍物，下面的路径数都为0
                break
            }
        }
    
        // 填充剩余的dp数组
        for i := 1; i < width; i++ {
            for j := 1; j < lenght; j++ {
                if obstacleGrid[i][j] == 1 {
                    dp[i][j] = 0
                    continue
                }
                dp[i][j] = dp[i-1][j] + dp[i][j-1]
            }
        }
    
        return dp[width-1][lenght-1]
    }

## 343.整数拆分

    func integerBreak(n int) int {
        dp:=make([]int,n+1)
        dp[0]=0
        dp[1]=0
        dp[2]=1
        for i:=3;i<n+1;i++{
            for j:=1;j<i;j++{
                dp[i]=max(max(j*(i-j),j*dp[i-j]),dp[i])
            }
        }
        return dp[n]
    }

每次比较的时候要注意，dp[i]的含义是第i个数的最大拆分值

## 96.不同的二叉搜索树

    func numTrees(n int) int {
        if n==1{
            return 1
        }
        dp:=make([]int,n+1)
        dp[0]=1
        dp[1]=1
        dp[2]=2
        for i:=3;i<n+1;i++{
            for j:=0;j<i;j++{
                dp[i] += dp[j]*dp[i-j-1]
            }
        }
        return dp[n]
    }

题目的含义就是 dp[i]等于每一个小于i的节点的所有情况相加
