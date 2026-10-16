/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <iostream>
#include <map>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include "include/BYTETracker.h"
#include "include/lapjv.h"

using std::cout;
using std::endl;
using std::map;
using std::max;
using std::min;
using std::pair;

vector<STrack*> BYTETracker::joint_stracks(vector<STrack*> &tlista, vector<STrack> &tlistb)
{
    map<int, int> exists;
    vector<STrack*> res;
    for (int i = 0; i < tlista.size(); i++)
    {
        exists.insert(pair<int, int>(tlista[i]->track_id, 1));
        res.push_back(tlista[i]);
    }
    for (int i = 0; i < tlistb.size(); i++)
    {
        int tid = tlistb[i].track_id;
        if (!exists[tid] || exists.count(tid) == 0)
        {
            exists[tid] = 1;
            res.push_back(&tlistb[i]);
        }
    }
    return res;
}

vector<STrack> BYTETracker::joint_stracks(vector<STrack> &tlista, vector<STrack> &tlistb)
{
    map<int, int> exists;
    vector<STrack> res;
    for (int i = 0; i < tlista.size(); i++)
    {
        exists.insert(pair<int, int>(tlista[i].track_id, 1));
        res.push_back(tlista[i]);
    }
    for (int i = 0; i < tlistb.size(); i++)
    {
        int tid = tlistb[i].track_id;
        if (!exists[tid] || exists.count(tid) == 0)
        {
            exists[tid] = 1;
            res.push_back(tlistb[i]);
        }
    }
    return res;
}

vector<STrack> BYTETracker::sub_stracks(vector<STrack> &tlista, vector<STrack> &tlistb)
{
    map<int, STrack> stracks;
    for (int i = 0; i < tlista.size(); i++)
    {
        stracks.insert(pair<int, STrack>(tlista[i].track_id, tlista[i]));
    }
    for (int i = 0; i < tlistb.size(); i++)
    {
        int tid = tlistb[i].track_id;
        if (stracks.count(tid) != 0)
        {
            stracks.erase(tid);
        }
    }

    vector<STrack> res;
    std::map<int, STrack>::iterator  it;
    for (it = stracks.begin(); it != stracks.end(); ++it)
    {
        res.push_back(it->second);
    }

    return res;
}

void BYTETracker::remove_duplicate_stracks(vector<STrack> &resa, vector<STrack> &resb,
    vector<STrack> &stracksa, vector<STrack> &stracksb)
{
    vector<vector<float> > pdist = iou_distance(stracksa, stracksb);
    vector<pair<int, int> > pairs;
    for (int i = 0; i < pdist.size(); i++)
    {
        for (int j = 0; j < pdist[i].size(); j++)
        {
            if (pdist[i][j] < 0.15)
            {
                pairs.push_back(pair<int, int>(i, j));
            }
        }
    }

    vector<int> dupa, dupb;
    for (int i = 0; i < pairs.size(); i++)
    {
        int timep = stracksa[pairs[i].first].frame_id - stracksa[pairs[i].first].start_frame;
        int timeq = stracksb[pairs[i].second].frame_id - stracksb[pairs[i].second].start_frame;
        if (timep > timeq)
            dupb.push_back(pairs[i].second);
        else
            dupa.push_back(pairs[i].first);
    }

    for (int i = 0; i < stracksa.size(); i++)
    {
        vector<int>::iterator iter = find(dupa.begin(), dupa.end(), i);
        if (iter == dupa.end())
        {
            resa.push_back(stracksa[i]);
        }
    }

    for (int i = 0; i < stracksb.size(); i++)
    {
        vector<int>::iterator iter = find(dupb.begin(), dupb.end(), i);
        if (iter == dupb.end())
        {
            resb.push_back(stracksb[i]);
        }
    }
}

void BYTETracker::linear_assignment(vector<vector<float> > &cost_matrix,
    int cost_matrix_size, int cost_matrix_size_size, float thresh,
    vector<vector<int> > &matches, vector<int> &unmatched_a,
    vector<int> &unmatched_b)
{
    if (cost_matrix.size() == 0)
    {
        for (int i = 0; i < cost_matrix_size; i++)
        {
            unmatched_a.push_back(i);
        }
        for (int i = 0; i < cost_matrix_size_size; i++)
        {
            unmatched_b.push_back(i);
        }
        return;
    }

    vector<int> rowsol; vector<int> colsol;
    float c = lapjv(cost_matrix, rowsol, colsol, true, thresh);
    for (int i = 0; i < rowsol.size(); i++)
    {
        if (rowsol[i] >= 0)
        {
            vector<int> match;
            match.push_back(i);
            match.push_back(rowsol[i]);
            matches.push_back(match);
        } else {
            unmatched_a.push_back(i);
        }
    }

    for (int i = 0; i < colsol.size(); i++)
    {
        if (colsol[i] < 0)
        {
            unmatched_b.push_back(i);
        }
    }
}

vector<vector<float> > BYTETracker::ious(vector<vector<float> > &atlbrs, vector<vector<float> > &btlbrs)
{
    vector<vector<float> > ious;
    if (atlbrs.size()*btlbrs.size() == 0)
        return ious;

    ious.resize(atlbrs.size());
    for (int i = 0; i < ious.size(); i++)
    {
        ious[i].resize(btlbrs.size());
    }

    // bbox_ious
    for (int k = 0; k < btlbrs.size(); k++)
    {
        vector<float> ious_tmp;
        float box_area = (btlbrs[k][2] - btlbrs[k][0] + 1)*(btlbrs[k][3] - btlbrs[k][1] + 1);
        for (int n = 0; n < atlbrs.size(); n++)
        {
            float iw = min(atlbrs[n][2], btlbrs[k][2]) - max(atlbrs[n][0], btlbrs[k][0]) + 1;
            if (iw > 0)
            {
                float ih = min(atlbrs[n][3], btlbrs[k][3]) - max(atlbrs[n][1], btlbrs[k][1]) + 1;
                if (ih > 0)
                {
                    float ua = (atlbrs[n][2] - atlbrs[n][0] + 1)*(atlbrs[n][3] - atlbrs[n][1] + 1) + box_area - iw * ih;
                    ious[n][k] = iw * ih / ua;
                } else {
                    ious[n][k] = 0.0;
                }
            } else {
                ious[n][k] = 0.0;
            }
        }
    }

    return ious;
}

vector<vector<float> > BYTETracker::iou_distance(vector<STrack*> &atracks,
    vector<STrack> &btracks, int &dist_size, int &dist_size_size)
{
    vector<vector<float> > cost_matrix;
    if (atracks.size() * btracks.size() == 0)
    {
        dist_size = atracks.size();
        dist_size_size = btracks.size();
        return cost_matrix;
    }
    vector<vector<float> > atlbrs, btlbrs;
    for (int i = 0; i < atracks.size(); i++)
    {
        atlbrs.push_back(atracks[i]->tlbr);
    }
    for (int i = 0; i < btracks.size(); i++)
    {
        btlbrs.push_back(btracks[i].tlbr);
    }

    dist_size = atracks.size();
    dist_size_size = btracks.size();

    vector<vector<float> > _ious = ious(atlbrs, btlbrs);

    for (int i = 0; i < _ious.size(); i++)
    {
        vector<float> _iou;
        for (int j = 0; j < _ious[i].size(); j++)
        {
            _iou.push_back(1 - _ious[i][j]);
        }
        cost_matrix.push_back(_iou);
    }

    return cost_matrix;
}

vector<vector<float> > BYTETracker::iou_distance(vector<STrack> &atracks, vector<STrack> &btracks)
{
    vector<vector<float> > atlbrs, btlbrs;
    for (int i = 0; i < atracks.size(); i++)
    {
        atlbrs.push_back(atracks[i].tlbr);
    }
    for (int i = 0; i < btracks.size(); i++)
    {
        btlbrs.push_back(btracks[i].tlbr);
    }

    vector<vector<float> > _ious = ious(atlbrs, btlbrs);
    vector<vector<float> > cost_matrix;
    for (int i = 0; i < _ious.size(); i++)
    {
        vector<float> _iou;
        for (int j = 0; j < _ious[i].size(); j++)
        {
            _iou.push_back(1 - _ious[i][j]);
        }
        cost_matrix.push_back(_iou);
    }

    return cost_matrix;
}

double BYTETracker::lapjv(const vector<vector<float> > &cost, vector<int> &rowsol, vector<int> &colsol,
    bool extend_cost, float cost_limit, bool return_cost)
{
    const size_t n_rows = cost.size();
    const size_t n_cols = cost.empty() ? 0 : cost.front().size();
    rowsol.assign(n_rows, -1);
    colsol.assign(n_cols, -1);
    for (const auto& row : cost)
    {
        if (row.size() != n_cols)
            throw std::invalid_argument("lapjv cost matrix must be rectangular");
    }
    if (n_rows == 0 || n_cols == 0)
        return 0.0;
    if (n_rows != n_cols && !extend_cost)
        throw std::invalid_argument("rectangular lapjv costs require extend_cost");

    const bool extended = extend_cost || cost_limit < LONG_MAX;
    const size_t n = extended ? n_rows + n_cols : n_rows;
    if (n > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        n > std::vector<double>().max_size() / n)
        throw std::length_error("lapjv cost matrix is too large");

    // Keep the original float padding values and double solver precision.
    float padding = 0.0f;
    if (extended)
    {
        if (cost_limit < LONG_MAX)
            padding = cost_limit / 2.0f;
        else
        {
            float cost_max = -1.0f;
            for (const auto& row : cost)
                for (float value : row)
                    cost_max = std::max(cost_max, value);
            padding = cost_max + 1.0f;
        }
    }
    // One contiguous allocation replaces over-sized per-row new[] buffers and
    // the two temporary float matrices. Row pointers serve the existing solver.
    std::vector<double> values(n * n, padding);
    std::vector<double*> row_ptrs(n);
    for (size_t i = 0; i < n; ++i)
    {
        row_ptrs[i] = values.data() + i * n;
        if (i < n_rows)
            std::copy(cost[i].begin(), cost[i].end(), row_ptrs[i]);
        else
            std::fill(row_ptrs[i] + n_cols, row_ptrs[i] + n, 0.0);
    }
    std::vector<int> x(n), y(n);
    if (lapjv_internal(static_cast<int>(n), row_ptrs.data(), x.data(), y.data()) != 0)
        throw std::runtime_error("lapjv assignment failed");

    double opt = 0.0;
    for (size_t i = 0; i < n_rows; ++i)
    {
        rowsol[i] = x[i] < static_cast<int>(n_cols) ? x[i] : -1;
        if (return_cost && rowsol[i] >= 0)
            opt += cost[i][rowsol[i]];
    }
    for (size_t j = 0; j < n_cols; ++j)
        colsol[j] = y[j] < static_cast<int>(n_rows) ? y[j] : -1;
    return opt;
}

Scalar BYTETracker::get_color(int idx)
{
    idx += 3;
    return Scalar(37 * idx % 255, 17 * idx % 255, 29 * idx % 255);
}
