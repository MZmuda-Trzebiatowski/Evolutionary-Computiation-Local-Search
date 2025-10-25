#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <cstdlib>
#include <climits>
#include <numeric>
#include <random>

using namespace std;

struct Node
{
    int x, y, cost;
};

struct Move
{
    int delta = INT_MAX;
    int type = -1; // 0: V-E, 1: Swap, 2: 2-opt
    int i = -1;
    int j = -1;
};

vector<Node> read_instance(const string &fname)
{
    ifstream in(fname);
    if (!in)
    {
        cerr << "Error: cannot open file " << fname << "\n";
        exit(1);
    }

    vector<Node> nodes;
    string line;
    while (getline(in, line))
    {
        if (line.empty())
            continue;

        for (char &c : line)
        {
            if (c == ';')
                c = ' ';
        }

        stringstream ss(line);
        int x, y, cost;
        if (!(ss >> x >> y >> cost))
        {
            cerr << "Warning: could not parse line -> " << line << "\n";
            continue;
        }
        nodes.push_back({x, y, cost});
    }

    return nodes;
}

vector<vector<int>> compute_distance_matrix(const vector<Node> &nodes)
{
    int n = nodes.size();
    vector<vector<int>> d(n, vector<int>(n, 0));
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            if (i == j)
            {
                d[i][j] = 0;
                continue;
            }
            double dx = nodes[i].x - nodes[j].x;
            double dy = nodes[i].y - nodes[j].y;
            double eu = sqrt(dx * dx + dy * dy);
            d[i][j] = (int)round(eu);
        }
    }
    return d;
}

int tour_objective(const vector<int> &tour, const vector<vector<int>> &d, const vector<Node> &nodes)
{
    int sum_d = 0;
    int m = tour.size();
    for (int i = 0; i < m; i++)
    {
        int a = tour[i];
        int b = tour[(i + 1) % m]; // wrap around
        sum_d += d[a][b];
    }
    int sum_cost = 0;
    for (int v : tour)
        sum_cost += nodes[v].cost;
    return sum_d + sum_cost;
}

// ------------------------------------ ASSIGNMENT 1&2 HEURISTICS ------------------------------------ //

vector<int> random_solution(int n, int k)
{
    vector<int> ids(n);
    iota(ids.begin(), ids.end(), 0);
    shuffle(ids.begin(), ids.end(), mt19937{random_device{}()});
    ids.resize(k);
    return ids;
}

vector<int> nn_end(int start_node, int k, const vector<vector<int>> &d, const vector<Node> &nodes)
{
    int n = d.size();
    vector<char> used(n, false);
    vector<int> path;

    path.reserve(k);
    path.push_back(start_node);
    used[start_node] = true;

    while ((int)path.size() < k)
    {
        int best_node = -1;
        int best_delta = INT_MAX;
        int iidx = path.back();

        for (int cand = 0; cand < n; cand++)
        {
            if (used[cand])
                continue;

            int delta = d[iidx][cand] + nodes[cand].cost;

            if (delta < best_delta)
            {
                best_delta = delta;
                best_node = cand;
            }
        }
        if (best_node == -1)
            break;
        path.push_back(best_node);
        used[best_node] = true;
    }
    return path;
}

vector<int> nn_anypos(int start_node, int k, const vector<vector<int>> &d, const vector<Node> &nodes)
{
    int n = d.size();
    vector<char> used(n, false);
    vector<int> path;
    path.reserve(k);

    path.push_back(start_node);
    used[start_node] = true;

    while ((int)path.size() < k)
    {
        int best_put_after_index = -1;
        int next_node = -1;
        int best_at_start_val = INT_MAX;

        // check for inserting at start
        for (int cand = 0; cand < n; cand++)
        {
            if (used[cand])
                continue;
            int val = d[cand][path[0]] + nodes[cand].cost;

            if (val < best_at_start_val)
            {
                best_at_start_val = val;
                next_node = cand;
            }
        }

        int best_nn = best_at_start_val;

        int c_size = (int)path.size();
        for (int i = 0; i < c_size; i++)
        {
            int best_next_node_locally = -1;
            int bnnl_min = INT_MAX;
            int put_after = path[i];
            int put_before = path[(i + 1) % c_size];

            for (int cand = 0; cand < n; cand++)
            {
                if (used[cand])
                    continue;
                int val = d[put_after][cand] + nodes[cand].cost + d[cand][put_before];
                if (i < ((i + 1) % c_size))
                {
                    val -= d[put_after][put_before];
                }

                if (val < bnnl_min)
                {
                    bnnl_min = val;
                    best_next_node_locally = cand;
                }
            }

            if (bnnl_min < best_nn)
            {
                best_nn = bnnl_min;
                next_node = best_next_node_locally;
                best_put_after_index = i;
            }
        }

        path.insert(path.begin() + best_put_after_index + 1, next_node);
        used[next_node] = true;
    }

    return path;
}

vector<int> greedy_cycle(int start_node, int k, const vector<vector<int>> &d, const vector<Node> &nodes)
{
    int n = d.size();
    vector<char> used(n, false);
    vector<int> cycle;
    cycle.reserve(k);

    cycle.push_back(start_node);
    used[start_node] = true;

    int best_second = -1;
    int best_delta = INT_MAX;

    // add second node to form first edge
    for (int cand = 0; cand < n; cand++)
    {
        if (used[cand])
            continue;
        int delta = d[start_node][cand] + nodes[cand].cost;
        if (delta < best_delta)
        {
            best_delta = delta;
            best_second = cand;
        }
    }

    cycle.push_back(best_second);
    used[best_second] = true;

    while ((int)cycle.size() < k)
    {
        int best_put_after_index = 0;
        int next_node = -1;
        int best_nn = INT_MAX;
        int c_size = (int)cycle.size();
        for (int i = 0; i < c_size; i++)
        {
            int best_next_node_locally = -1;
            int bnnl_min = INT_MAX;
            int put_after = cycle[i];
            int put_before = cycle[(i + 1) % c_size];

            for (int cand = 0; cand < n; cand++)
            {
                if (used[cand])
                    continue;
                int val = d[put_after][cand] + nodes[cand].cost + d[cand][put_before];
                if (c_size > 2)
                {
                    val -= d[put_after][put_before];
                }

                if (val < bnnl_min)
                {
                    bnnl_min = val;
                    best_next_node_locally = cand;
                }
            }

            if (bnnl_min < best_nn)
            {
                best_nn = bnnl_min;
                next_node = best_next_node_locally;
                best_put_after_index = i;
            }
        }

        cycle.insert(cycle.begin() + best_put_after_index + 1, next_node);
        used[next_node] = true;
    }

    return cycle;
}

vector<int> nn_anypos_regret(int start_node, int k, const vector<vector<int>> &d, const vector<Node> &nodes, double w1, double w2)
{
    struct RegretRankingEntry
    {
        int node;
        int first_pos;
        int first_val;
        int second_pos;
        int second_val;
    };

    int n = d.size();
    vector<char> used(n, false);
    vector<int> path;
    path.reserve(k);

    path.push_back(start_node);
    used[start_node] = true;

    while ((int)path.size() < k)
    {

        vector<RegretRankingEntry> ranking;

        for (int i = 0; i < n; i++)
        {
            if (used[i])
                continue;

            int best_first_pos = 0;
            int best_first_val = d[i][path[0]] + nodes[i].cost;
            int best_second_pos = 0;
            int best_second_val = INT_MAX;

            int c_size = (int)path.size();
            for (int pos = 1; pos < c_size; pos++)
            {
                int val = d[path[pos - 1]][i] + d[path[pos]][i] - d[path[pos - 1]][path[pos]] + nodes[i].cost;

                if (val < best_first_val)
                {
                    best_second_val = best_first_val;
                    best_second_pos = best_first_pos;
                    best_first_val = val;
                    best_first_pos = pos;
                }
                else if (val < best_second_val)
                {
                    best_second_val = val;
                    best_second_pos = pos;
                }
            }

            int end_val = d[path[c_size - 1]][i] + nodes[i].cost;
            if (end_val < best_first_val)
            {
                best_second_val = best_first_val;
                best_second_pos = best_first_pos;
                best_first_val = end_val;
                best_first_pos = c_size;
            }
            else if (end_val < best_second_val)
            {
                best_second_val = end_val;
                best_second_pos = c_size;
            }

            ranking.push_back({i, best_first_pos, best_first_val, best_second_pos, best_second_val});
        }

        double best_score = -__DBL_MAX__;
        int best_node = 0;
        int best_pos = 0;

        for (RegretRankingEntry entry : ranking)
        {
            double score = w1 * (double)(entry.second_val - entry.first_val) - w2 * (double)entry.first_val;

            if (score > best_score)
            {
                best_score = score;
                best_node = entry.node;
                best_pos = entry.first_pos;
            }
        }

        path.insert(path.begin() + best_pos, best_node);
        used[best_node] = true;
    }

    return path;
}

vector<int> greedy_cycle_regret(int start_node, int k, const vector<vector<int>> &d, const vector<Node> &nodes, double w1, double w2)
{
    struct RegretRankingEntry
    {
        int node;
        int first_pos;
        int first_val;
        int second_pos;
        int second_val;
    };

    int n = d.size();
    vector<char> used(n, false);
    vector<int> cycle;
    cycle.reserve(k);

    cycle.push_back(start_node);
    used[start_node] = true;

    int best_second = -1;
    int best_delta = INT_MAX;

    // add second node to form first edge
    for (int cand = 0; cand < n; cand++)
    {
        if (used[cand])
            continue;
        int delta = d[start_node][cand] + nodes[cand].cost;
        if (delta < best_delta)
        {
            best_delta = delta;
            best_second = cand;
        }
    }

    cycle.push_back(best_second);
    used[best_second] = true;

    while ((int)cycle.size() < k)
    {

        vector<RegretRankingEntry> ranking;

        for (int i = 0; i < n; i++)
        {
            if (used[i])
                continue;

            int best_first_pos = 0;
            int best_first_val = INT_MAX;
            int best_second_pos = 0;
            int best_second_val = INT_MAX;

            int c_size = (int)cycle.size();
            for (int pos = 1; pos <= c_size; pos++)
            {
                int val = d[cycle[pos - 1]][i] + d[cycle[pos % c_size]][i] + nodes[i].cost;
                if (c_size > 2)
                {
                    val -= d[cycle[pos - 1]][cycle[pos % c_size]];
                }

                if (val < best_first_val)
                {
                    best_second_val = best_first_val;
                    best_second_pos = best_first_pos;
                    best_first_val = val;
                    best_first_pos = pos;
                }
                else if (val < best_second_val)
                {
                    best_second_val = val;
                    best_second_pos = pos;
                }
            }

            ranking.push_back({i, best_first_pos, best_first_val, best_second_pos, best_second_val});
        }

        double best_score = -__DBL_MAX__;
        int best_node = 0;
        int best_pos = 0;

        for (RegretRankingEntry entry : ranking)
        {
            double score = w1 * (double)(entry.second_val - entry.first_val) - w2 * (double)entry.first_val;

            if (score > best_score)
            {
                best_score = score;
                best_node = entry.node;
                best_pos = entry.first_pos;
            }
        }

        cycle.insert(cycle.begin() + best_pos, best_node);
        used[best_node] = true;
    }

    return cycle;
}

// ------------------------------------ HELPERS ------------------------------------ //

void export_tour_svg(const string &filename, const vector<int> &tour, const vector<Node> &nodes)
{
    if (tour.empty())
    {
        cerr << "Warning: Cannot export empty tour.\n";
        return;
    }

    // 1. Determine bounding box for scaling
    int min_x = nodes[0].x, max_x = nodes[0].x;
    int min_y = nodes[0].y, max_y = nodes[0].y;
    int min_cost = nodes[0].cost, max_cost = nodes[0].cost;

    for (const auto &node : nodes)
    {
        min_x = min(min_x, node.x);
        max_x = max(max_x, node.x);
        min_y = min(min_y, node.y);
        max_y = max(max_y, node.y);
        min_cost = min(min_cost, node.cost);
        max_cost = max(max_cost, node.cost);
    }

    // 2. Define SVG canvas parameters
    const int SVG_WIDTH = 1920;
    const int SVG_HEIGHT = 1080;
    const int PADDING = 40; // Space from the edge

    double scale_x = (double)(SVG_WIDTH - 2 * PADDING) / (max_x - min_x + 1);
    double scale_y = (double)(SVG_HEIGHT - 2 * PADDING) / (max_y - min_y + 1);
    double scale = min(scale_x, scale_y); // Use the smaller scale factor to maintain aspect ratio

    // Function to scale coordinates
    auto scale_coord_x = [&](int x)
    {
        return PADDING + (x - min_x) * scale;
    };
    auto scale_coord_y = [&](int y)
    {
        // SVG y-axis is top-down, so we invert the scaling
        return SVG_HEIGHT - PADDING - (y - min_y) * scale;
    };

    // Function to scale cost to radius (min radius 3, max radius 15)
    auto scale_cost_to_radius = [&](int cost)
    {
        if (max_cost == min_cost)
            return 6.0;
        double normalized = (double)(cost - min_cost) / (max_cost - min_cost);
        return 3.0 + normalized * 12.0; // Radius between 3 and 15
    };

    // 3. Open file and write SVG header
    ofstream out(filename);
    if (!out)
    {
        cerr << "Error: cannot create SVG file " << filename << "\n";
        return;
    }

    out << "<svg width=\"" << SVG_WIDTH << "\" height=\"" << SVG_HEIGHT << "\" xmlns=\"http://www.w3.org/2000/svg\">\n";
    // Background color
    out << "  <rect width=\"100%\" height=\"100%\" fill=\"#f8f8f8\"/>\n";
    // Text label for the best objective value
    auto d_matrix = compute_distance_matrix(nodes); // Recompute or pass d
    int obj = tour_objective(tour, d_matrix, nodes);
    out << "  <text x=\"" << PADDING << "\" y=\"" << PADDING / 2.0 << "\" font-family=\"sans-serif\" font-size=\"14\" fill=\"#333\">\n";
    out << "    Best Objective: " << obj << " | Tour Size: " << tour.size() << "\n";
    out << "  </text>\n";

    // 4. Draw edges (lines)
    out << "  \n";
    int m = tour.size();
    for (int i = 0; i < m; ++i)
    {
        int idx1 = tour[i];
        int idx2 = tour[(i + 1) % m];

        double x1 = scale_coord_x(nodes[idx1].x);
        double y1 = scale_coord_y(nodes[idx1].y);
        double x2 = scale_coord_x(nodes[idx2].x);
        double y2 = scale_coord_y(nodes[idx2].y);

        out << "  <line x1=\"" << x1 << "\" y1=\"" << y1 << "\" x2=\"" << x2 << "\" y2=\"" << y2 << "\"\n";
        out << "        stroke=\"#0000FF\" stroke-width=\"2\" stroke-dasharray=\"4,2\"/>\n";
    }

    // 5. Draw nodes (circles)
    out << "  \n";
    int start_node_idx = tour[0]; // Get the index of the first node in the tour

    for (size_t i = 0; i < nodes.size(); ++i)
    {
        int current_x = nodes[i].x;
        int current_y = nodes[i].y;
        int current_cost = nodes[i].cost;

        double cx = scale_coord_x(current_x);
        double cy = scale_coord_y(current_y);
        double r = scale_cost_to_radius(current_cost);

        // Check if the node is in the tour
        bool in_tour = (find(tour.begin(), tour.end(), i) != tour.end());

        string fill_color;
        string stroke_color;
        double stroke_width = 1.5; // Default stroke width

        if (i == start_node_idx)
        {                             // This is the starting node
            fill_color = "#00AA00";   // Green color
            stroke_color = "#006400"; // Darker green border
            stroke_width = 3.0;       // Thicker border for start node
        }
        else if (in_tour)
        {                             // Other selected nodes
            fill_color = "#FF0000";   // Red color
            stroke_color = "#8B0000"; // Darker red border
        }
        else
        {                             // Unselected nodes
            fill_color = "#AAAAAA";   // Grey color
            stroke_color = "#666666"; // Darker grey border
        }

        out << "  <circle cx=\"" << cx << "\" cy=\"" << cy << "\" r=\"" << r << "\"\n";
        out << "          fill=\"" << fill_color << "\" stroke=\"" << stroke_color << "\" stroke-width=\"" << stroke_width << "\">\n";
        out << "    <title>Node " << i << " (x:" << current_x << ", y:" << current_y << ", cost:" << current_cost << (i == start_node_idx ? ", START" : "") << ")</title>\n";
        out << "  </circle>\n";
    }

    // 6. Close SVG tag
    out << "</svg>\n";
    out.close();

    // cout << "SVG visualization saved to: " << filename << "\n";
}

void print_stats(const string &heuristic_name, const vector<int> &objectives)
{
    if (objectives.empty())
    {
        cerr << "Error: No objectives recorded for " << heuristic_name << ".\n";
        return;
    }

    int min_obj = *min_element(objectives.begin(), objectives.end());
    int max_obj = *max_element(objectives.begin(), objectives.end());

    long double sum_obj = accumulate(objectives.begin(), objectives.end(), (long double)0.0);
    long double avg_obj = sum_obj / objectives.size();

    cout << "\n " << heuristic_name << " Stats\n";
    cout << "  Min Objective: " << min_obj << "\n";
    cout << "  Max Objective: " << max_obj << "\n";
    cout << "  Avg Objective: " << avg_obj << "\n";
    cout << "------------------------------------------\n";
}

void export_tour_txt(const string &filename, const vector<int> &tour)
{
    ofstream outfile(filename);
    if (outfile.is_open())
    {
        for (size_t i = 0; i < tour.size(); ++i)
        {
            outfile << tour[i];
            if (i < tour.size() - 1)
            {
                outfile << ", ";
            }
        }
        outfile << "\n";
        outfile.close();
        cout << "  > Exported best tour indices to " << filename << " (TXT file).\n";
    }
    else
    {
        cerr << "  > ERROR: Unable to open file " << filename << " for writing.\n";
    }
}

// ------------------------------------ ASSIGNMENT 3 LOCAL SEARCH ------------------------------------ //

// --- Delta Calculation Functions ---

// Inter-route: Swap selected node v_idx with unselected node e_idx
int delta_V_E_exchange(const vector<int> &tour, int v_idx, int e_idx, const vector<vector<int>> &d, const vector<Node> &nodes)
{
    int m = tour.size();
    int v = tour[v_idx]; // The node to remove

    int prev_v_idx = (v_idx + m - 1) % m;
    int next_v_idx = (v_idx + 1) % m;
    int prev_v = tour[prev_v_idx];
    int next_v = tour[next_v_idx];

    int dist_delta = (d[prev_v][e_idx] + d[e_idx][next_v]) - (d[prev_v][v] + d[v][next_v]);

    int cost_delta = nodes[e_idx].cost - nodes[v].cost;

    return dist_delta + cost_delta;
}

// Intra-route: Swap two selected nodes v_i and v_j
int delta_swap(const vector<int> &tour, int i, int j, const vector<vector<int>> &d)
{
    if (i == j) return INT_MAX;
    int m = tour.size();

    if (i > j) std::swap(i, j);

    int v_i = tour[i];
    int v_j = tour[j];

    int prev_i = tour[(i + m - 1) % m];
    int next_i = tour[(i + 1) % m];

    int prev_j = tour[(j + m - 1) % m];
    int next_j = tour[(j + 1) % m];

    int dist_delta = 0;

    if (j == (i + 1) % m) // adjacent nodes
    {
        dist_delta = (d[prev_i][v_j] + d[v_j][v_i] + d[v_i][next_j]) - 
                     (d[prev_i][v_i] + d[v_i][v_j] + d[v_j][next_j]);
    }
    else // non-adjacent nodes
    {
        dist_delta = (d[prev_i][v_j] + d[v_j][next_i] + d[prev_j][v_i] + d[v_i][next_j]) - 
                     (d[prev_i][v_i] + d[v_i][next_i] + d[prev_j][v_j] + d[v_j][next_j]);
    }

    return dist_delta;
}

// Intra-route: 2-opt, exchange edges (i, i+1) and (j, j+1)
int delta_2opt(const vector<int> &tour, int i, int j, const vector<vector<int>> &d)
{
    int m = tour.size();
    
    int i_plus_1 = (i + 1) % m;
    int j_plus_1 = (j + 1) % m;

    if (i == j || i_plus_1 == j || j_plus_1 == i) return INT_MAX;
    
    // Edges (i, i+1) and (j, j+1) are replaced by (i, j) and (i+1, j+1)
    int v_i = tour[i];
    int v_i_plus_1 = tour[i_plus_1];
    int v_j = tour[j];
    int v_j_plus_1 = tour[j_plus_1];
    
    int dist_delta = (d[v_i][v_j] + d[v_i_plus_1][v_j_plus_1]) - (d[v_i][v_i_plus_1] + d[v_j][v_j_plus_1]);

    return dist_delta;
}

// --- Apply Move Functions ---

void apply_V_E_exchange(vector<int> &tour, int v_idx, int e_idx)
{
    tour[v_idx] = e_idx;
}

void apply_swap(vector<int> &tour, int i, int j)
{
    swap(tour[i], tour[j]);
}

void apply_2opt(vector<int> &tour, int i, int j)
{
    int m = tour.size();
    int i_plus_1 = (i + 1) % m;

    if (i_plus_1 == 0 && j == m - 1) return; // No change if full reversal

    if (i_plus_1 <= j) // Simple case: reverse segment between i+1 and j
    {
        reverse(tour.begin() + i_plus_1, tour.begin() + j + 1);
    }
    else
    {
        vector<int> segment;
        for (int k = i_plus_1; k < m; ++k) segment.push_back(tour[k]);
        for (int k = 0; k <= j; ++k) segment.push_back(tour[k]);
        
        reverse(segment.begin(), segment.end());

        // Re-insert reversed segment
        int current = 0;
        for (int k = i_plus_1; k < m; ++k) tour[k] = segment[current++];
        for (int k = 0; k <= j; ++k) tour[k] = segment[current++];
    }
}

// ------------------------------------- MAIN FUNCTION ------------------------------------ //
int main(int argc, char **argv)
{
    
}