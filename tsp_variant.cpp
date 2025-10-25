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
#include <chrono>

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

void print_stats(const string &heuristic_name, const vector<int> &objectives, double total_time_ms)
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
    
    // Time statistics
    long double avg_time_ms = total_time_ms / objectives.size();

    cout << "\n=========================================================\n";
    cout << " " << heuristic_name << " Stats\n";
    cout << "=========================================================\n";
    cout << "  Min Objective: " << min_obj << "\n";
    cout << "  Max Objective: " << max_obj << "\n";
    cout << "  Avg Objective: " << avg_obj << "\n";
    cout << "------------------------------------------\n";
    cout << "  Total Running Time: " << total_time_ms / 1000.0 << " s\n";
    cout << "  Avg Time per Run: " << avg_time_ms << " ms\n";
    cout << "=========================================================\n";
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
        // cout << "  > Exported best tour indices to " << filename << " (TXT file).\n";
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
        if (i == j) return 0;
        int m = tour.size();

        if (i > j) swap(i, j);

        int v_i = tour[i];
        int v_j = tour[j];

        int dist_delta = 0;

        if (abs(i - j) == 1) // adjacent nodes
        {
            int prev = (i - 1 + m) % m;
            int next = (j + 1) % m;
            dist_delta = (d[tour[prev]][v_j]  + d[v_i][tour[next]])
                        - (d[tour[prev]][v_i] + d[v_j][tour[next]]);
        }
        else if ((i == 0 && j == m - 1))
        {
            int prev = (j - 1 + m) % m;
            int next = (i + 1) % m;
            dist_delta = (d[tour[prev]][v_i]  + d[v_j][tour[next]])
                        - (d[tour[prev]][v_j] + d[v_i][tour[next]]);
        }
        else // non-adjacent nodes
        {
            int prev_i = tour[(i + m - 1) % m];
            int next_i = tour[(i + 1) % m];

            int prev_j = tour[(j + m - 1) % m];
            int next_j = tour[(j + 1) % m];

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

// --- Local Search Functions ---

// Local Search steepest
vector<int> local_search_steepest(vector<int> tour, const vector<vector<int>> &d, const vector<Node> &nodes, bool use_swap_intra)
{
    int n = d.size();
    int k = tour.size();
    
    mt19937 rng{random_device{}()};

    vector<bool> is_selected(n, false);
    for(int v : tour) is_selected[v] = true;
    
    while (true)
    {
        Move best_move;
        best_move.delta = 0;
        
        // 1. Inter-route: V-E Exchange
        for (int i = 0; i < k; ++i)
        {
            for (int j = 0; j < n; ++j) 
            {
                if (is_selected[j]) continue;
                
                int delta = delta_V_E_exchange(tour, i, j, d, nodes);
                
                if (delta < best_move.delta)
                {
                    best_move.delta = delta;
                    best_move.type = 0; // V-E
                    best_move.i = i; 
                    best_move.j = j; 
                }
            }
        }
 
        // 2. Intra-route: Swap OR 2-opt
        for (int i = 0; i < k; ++i)
        {
            for (int j = i + 1; j < k; ++j)
            {
                int delta = INT_MAX;
                int move_type = -1;

                if (use_swap_intra)
                {
                    delta = delta_swap(tour, i, j, d);
                    move_type = 1; // Swap
                }
                else
                {
                    delta = delta_2opt(tour, i, j, d);
                    move_type = 2; // 2-opt
                }
                
                if (delta < best_move.delta)
                {
                    best_move.delta = delta;
                    best_move.type = move_type;
                    best_move.i = i;
                    best_move.j = j;
                }
            }
        }

        if (best_move.delta >= 0)
        {
            break;
        }

        if (best_move.type == 0)
        {
            int removed_node = tour[best_move.i];
            apply_V_E_exchange(tour, best_move.i, best_move.j);
            is_selected[removed_node] = false;
            is_selected[best_move.j] = true;
        }
        else if (best_move.type == 1)
        {
            apply_swap(tour, best_move.i, best_move.j);
        }
        else if (best_move.type == 2)
        {
            apply_2opt(tour, best_move.i, best_move.j);
        }
        // cout << "applied move type " << best_move.type << " with delta " << best_move.delta << " moves: " << best_move.i << ", " << best_move.j << "\n";
    }
    return tour;
}

// Local Search greedy
vector<int> local_search_greedy(vector<int> tour, const vector<vector<int>> &d, const vector<Node> &nodes, bool use_swap_intra)
{
    int n = d.size();
    int k = tour.size();

    mt19937 rng(random_device{}());

    vector<bool> is_selected(n, false);
    for(int v : tour) is_selected[v] = true;

    while (true)
    {
        bool improved_in_iteration = false;

        // Setup indices for random iteration over tour nodes
        vector<int> tour_indices(k);
        iota(tour_indices.begin(), tour_indices.end(), 0);

        vector<int> unselected_nodes;
        for (int i = 0; i < n; ++i)
        {
            if (!is_selected[i])
                unselected_nodes.push_back(i);
        }

        // Random order of neighborhood types (0: Inter, 1: Intra)
        vector<int> neighborhood_order = {0, 1};
        shuffle(neighborhood_order.begin(), neighborhood_order.end(), rng);

        for (int move_type_code : neighborhood_order)
        {
            if (move_type_code == 0)
            {
                shuffle(tour_indices.begin(), tour_indices.end(), rng); // Random order for selected node
                shuffle(unselected_nodes.begin(), unselected_nodes.end(), rng); // Random order for unselected node

                for (int i : tour_indices)
                {
                    for (int u : unselected_nodes)
                    {
                        int delta = delta_V_E_exchange(tour, i, u, d, nodes);

                        if (delta < 0)
                        {
                            // Found the first improving move (Greedy Acceptance)
                            int removed_node = tour[i];
                            apply_V_E_exchange(tour, i, u); 
                            
                            is_selected[removed_node] = false;
                            is_selected[u] = true;
                            
                            improved_in_iteration = true;
                            goto next_iteration; 
                        }
                    }
                }
            }
            else 
            {
                shuffle(tour_indices.begin(), tour_indices.end(), rng); // Random order for 'i'

                for (int i : tour_indices)
                {
                    vector<int> j_indices;
                    for (int l = i + 1; l < k; ++l)
                        j_indices.push_back(l);
                    shuffle(j_indices.begin(), j_indices.end(), rng);

                    for (int j : j_indices)
                    {
                        int delta = INT_MAX;
                        int applied_type = -1;

                        if (use_swap_intra)
                        {
                            delta = delta_swap(tour, i, j, d);
                            applied_type = 1; // Swap
                        }
                        else
                        {
                            delta = delta_2opt(tour, i, j, d);
                            applied_type = 2; // 2-opt
                        }

                        if (delta < 0)
                        {
                            // Found the first improving move (Greedy Acceptance)
                            if (applied_type == 1)
                            {
                                apply_swap(tour, i, j);
                            }
                            else if (applied_type == 2)
                            {
                                apply_2opt(tour, i, j);
                            }

                            improved_in_iteration = true;
                            goto next_iteration; // Found improvement, break inner loops and restart main loop
                        }
                    }
                }
                
            }
        }

    next_iteration:; // Label for breaking out of nested loops

        // Check the stopping condition
        if (!improved_in_iteration)
        {
            // No improving move was found in the complete iteration over all neighborhoods
            break; 
        }
    }
    return tour;
}

// --- Experiment Helper functions ---
vector<int> get_greedy_start(int start_node, int k, const vector<vector<int>> &d, const vector<Node> &nodes)
{
    return nn_anypos_regret(start_node, k, d, nodes, 0.5, 0.5);
}

void run_local_search_experiment(const string& name, bool steepest, bool use_swap, bool random_start, int n, int k, const vector<vector<int>> &d, const vector<Node> &nodes)
{
    const int N_RUNS = 200;
    vector<int> objectives;
    int best_obj = INT_MAX;
    vector<int> best_tour;

    mt19937 rand_start_rng{6767};

    auto total_start = chrono::high_resolution_clock::now();

    for (int t = 0; t < N_RUNS; t++)
    {
        vector<int> start_tour;
        if (random_start)
        {
            start_tour = random_solution(n, k);

        }
        else
        {
            int start_node = t % n;
            start_tour = get_greedy_start(start_node, k, d, nodes);
        }

        vector<int> final_tour;
        if (steepest)
        {
            final_tour = local_search_steepest(start_tour, d, nodes, use_swap);
        }
        else
        {
            final_tour = local_search_greedy(start_tour, d, nodes, use_swap);
        }

        int obj = tour_objective(final_tour, d, nodes);
        objectives.push_back(obj);

        if (obj < best_obj)
        {
            best_obj = obj;
            best_tour = final_tour;
        }
    }

    auto total_end = chrono::high_resolution_clock::now();
    auto total_duration = chrono::duration_cast<chrono::milliseconds>(total_end - total_start);
    double total_time_ms = total_duration.count();
    
    string ls_type = steepest ? "Steepest" : "Greedy";
    string intra_type = use_swap ? "Swap" : "2-opt";
    string start_type = random_start ? "Random_INIT" : "Greedy_INIT";
    
    string full_name = ls_type + "_" + intra_type + "_" + start_type;
    
    print_stats("Local Search (" + full_name + ")", objectives, total_time_ms);
    export_tour_svg("best_ls_" + full_name + "_tour.svg", best_tour, nodes);
    export_tour_txt("best_ls_" + full_name + "_tour.txt", best_tour);
}

// ------------------------------------- MAIN FUNCTION ------------------------------------ //
int main(int argc, char **argv)
{
    if (argc < 2)
    {
        cerr << "Usage: " << argv[0] << " <instance-file>\n";
        return 1;
    }
    string fname = argv[1];
    vector<Node> nodes = read_instance(fname);
    int n = nodes.size();
    if (n == 0)
    {
        cerr << "No nodes read from file.\n";
        return 1;
    }

    int k = (n + 1) / 2; // half of the nodes, rounded up
    cout << "Read " << n << " nodes; selecting k = " << k << " nodes per solution.\n";

    auto d = compute_distance_matrix(nodes);

    for (int ls_type = 0; ls_type < 2; ++ls_type) // 0: Steepest, 1: Greedy
    {
        bool steepest = (ls_type == 0);
        
        for (int intra_type = 0; intra_type < 2; ++intra_type) // 0: Swap, 1: 2-opt
        {
            bool use_swap = (intra_type == 0);
            
            for (int start_type = 0; start_type < 2; ++start_type) // 0: Random, 1: Greedy
            {
                bool random_start = (start_type == 0);
                
                run_local_search_experiment("", steepest, use_swap, random_start, n, k, d, nodes);
            }
        }
    }
}