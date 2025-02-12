#include <iostream>
#include <vector>
#include <limits>
#include <cmath>
#include <algorithm>
#include <random>
#include <ctime>
#include <iomanip>
#include <omp.h>

class KDNode
{
public:
    std::vector<float> point;
    KDNode *left, *right;
    int axis;

    KDNode(const std::vector<float> &point, int axis, KDNode *left = nullptr, KDNode *right = nullptr) : point(point), axis(axis), left(left), right(right) {}

    ~KDNode()
    {
        delete left;
        delete right;
    }

    void insert(const std::vector<float> &new_point)
    {
        int next_axis = (this->axis + 1) % this->point.size();
        if (new_point[this->axis] < this->point[this->axis])
        {
            if (!this->left)
            {
                this->left = new KDNode(new_point, next_axis);
            }
            else
            {
                this->left->insert(new_point);
            }
        }
        else
        {
            if (!this->right)
            {
                this->right = new KDNode(new_point, next_axis);
            }
            else
            {
                this->right->insert(new_point);
            }
        }
    }

    void range_search(const std::vector<float> &lower, const std::vector<float> &upper, std::vector<KDNode *> &results)
    {
        if (!this)
            return;

        bool inside = true;
        for (size_t i = 0; i < point.size(); i++)
        {
            if (point[i] < lower[i] || point[i] > upper[i])
            {
                inside = false;
                break;
            }
        }

        if (inside)
            results.push_back(this);

        if (this->left && this->point[this->axis] >= lower[this->axis])
        {
            this->left->range_search(lower, upper, results);
        }
        if (this->right && this->point[this->axis] <= upper[this->axis])
        {
            this->right->range_search(lower, upper, results);
        }
    }

    static float squared_distance(const std::vector<float> &p1, const std::vector<float> &p2)
    {
        float dist = 0.0;
        for (size_t i = 0; i < p1.size(); i++)
        {
            dist += (p1[i] - p2[i]) * (p1[i] - p2[i]);
        }
        return dist;
    }

    KDNode *find_nearest(const std::vector<float> &target_point)
    {
        return find_nearest(this, target_point, this, squared_distance(this->point, target_point));
    }

    static void print_tree(KDNode *node, int depth = 0)
    {
        if (!node)
            return;
        print_tree(node->right, depth + 1);
        std::cout << std::string(depth * 4, ' ') << "(";
        for (size_t i = 0; i < node->point.size(); i++)
        {
            std::cout << node->point[i] << (i < node->point.size() - 1 ? ", " : "");
        }
        std::cout << ")" << std::endl;
        print_tree(node->left, depth + 1);
    }

    void print_tree(int depth = 0) const
    {
        if (right != nullptr)
        {
            right->print_tree(depth + 4);
        }
        std::cout << std::setw(depth) << " " << "(";
        for (size_t i = 0; i < point.size(); i++)
        {
            std::cout << point[i] << (i < point.size() - 1 ? ", " : "");
        }
        std::cout << ")" << std::endl;
        if (left != nullptr)
        {
            left->print_tree(depth + 4);
        }
    }

    static int tree_size(KDNode *node)
    {
        if (!node)
            return 0;
        return 1 + tree_size(node->left) + tree_size(node->right);
    }

private:
    static KDNode *find_nearest(KDNode *node, const std::vector<float> &target_point, KDNode *best, float best_dist)
    {
        if (!node)
            return best;

        float dist = squared_distance(target_point, node->point);
        if (dist < best_dist)
        {
            best_dist = dist;
            best = node;
        }

        int next_axis = (node->axis + 1) % node->point.size();
        KDNode *next_node = (target_point[node->axis] < node->point[node->axis]) ? node->left : node->right;
        KDNode *other_node = (next_node == node->left) ? node->right : node->left;

        #pragma omp parallel sections
        {
            #pragma omp section
            {
                best = find_nearest(next_node, target_point, best, best_dist);
            }
            #pragma omp section
            {
                float axis_dist = (target_point[node->axis] - node->point[node->axis]) * (target_point[node->axis] - node->point[node->axis]);
                if (axis_dist < best_dist)
                {
                    best = find_nearest(other_node, target_point, best, best_dist);
                }
            }
        }

        return best;
    }
};

int main()
{
    std::srand(static_cast<unsigned int>(std::time(nullptr)));

    // Generating a k-d tree
    int num_points = 1000; // Number of points in the k-d tree
    int dimensions = 2;    // Number of dimensions
    KDNode *root = nullptr;
    for (int i = 0; i < num_points; ++i)
    {
        std::vector<float> point(dimensions);
        for (int d = 0; d < dimensions; ++d)
        {
            point[d] = static_cast<float>(std::rand() % 1000) / 100.0;
        }
        if (i == 0)
        {
            root = new KDNode(point, 0);
        }
        else
        {
            root->insert(point);
        }
    }

    std::vector<float> target(dimensions);
    for (int d = 0; d < dimensions; ++d)
    {
        target[d] = static_cast<float>(std::rand() % 1000) / 100.0;
    }

    KDNode *nearest = root->find_nearest(target);
    if (nearest)
    {
        std::cout << "Nearest point to (";
        for (float v : target)
        {
            std::cout << v << " ";
        }
        std::cout << "): ";
        for (float v : nearest->point)
        {
            std::cout << v << " ";
        }
        std::cout << std::endl;
    }

    std::vector<KDNode *> range_results;
    std::vector<float> lower(dimensions, 20.0f);
    std::vector<float> upper(dimensions, 50.0f);
    root->range_search(lower, upper, range_results);
    std::cout << "Range search results (points between [20.0, 50.0]): " << std::endl;
    for (auto node : range_results)
    {
        for (float v : node->point)
        {
            std::cout << v << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "Size of KD-Tree: " << KDNode::tree_size(root) << std::endl;

    delete root;

    KDNode *root2 = new KDNode({3.0, 1.5}, 0);
    root2->insert({2.0, 3.0});
    root2->insert({4.0, 2.0});
    root2->insert({4.0, 2.5});
    root2->insert({4.5, 1.0});
    root2->insert({4.5, 1.5});
    root2->insert({2.0, 1.0});

    std::cout << "KD-Tree structure:" << std::endl;
    root2->print_tree();

    delete root2;

    return 0;
}
