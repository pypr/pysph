#ifndef SPATIAL_HASH_H
#define SPATIAL_HASH_H
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>

 // p1, p2 and p3 are large primes used in the hash function
 // Ref. http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.105.6732&rep=rep1&type=pdf

#define p1 73856093
#define p2 19349663
#define p3 83492791

using namespace std;

class HashEntry
{
private:
    long long int key;
public:
    int c_x, c_y, c_z;
    HashEntry* next;
    vector <unsigned int> indices;
    HashEntry(long long int key, int idx, int c_x, int c_y, int c_z)
    {
        this->key = key;
        this->indices.push_back(idx);
        this->c_x = c_x;
        this->c_y = c_y;
        this->c_z = c_z;
        this->next = NULL;
    }

    inline long long int get_key()
    {
        return this->key;
    }

    inline vector <unsigned int> *get_indices()
    {
        return &this->indices;
    } 
};

class HashTable
{
private:
    HashEntry** hashtable;
public:
    long long int table_size;
    
    HashTable(long long int table_size)
    {
        this->table_size = table_size;
        this->hashtable = new HashEntry*[table_size];
        for(int i=0; i<table_size; i++)
        {
            this->hashtable[i] = NULL;
        }
    }

    inline long long int hash(long long int i, long long int j, long long int k)
    {
        return ((i*p1)^(j*p2)^(k*p3))%this->table_size;
    }

    void add(int i, int j, int k, int idx)
    {
        long long int key = this->hash(i,j,k);
        HashEntry* prev = NULL;
        HashEntry* entry = this->hashtable[key];
        while(entry!=NULL)
        {
            if(entry->c_x==i && entry->c_y==j && entry->c_z==k)
                break;
            prev = entry;
            entry = entry->next;
        }
        if(entry!=NULL)
            entry->indices.push_back(idx);
        else
        {
            entry = new HashEntry(key, idx, i, j, k);
            if(prev==NULL)
                this->hashtable[key] = entry;
            else
                prev->next = entry;
        }
    }

    vector <unsigned int> *get(int i, int j, int k)
    {
        long long int key = this->hash(i,j,k);
        HashEntry* entry = this->hashtable[key];
        while(entry!=NULL)
        {
            if(entry->c_x==i && entry->c_y==j && entry->c_z==k)
                return entry->get_indices();
            entry = entry->next;
        }
        return NULL;
    }

    ~HashTable()
    {
        for(int i=0; i<this->table_size; i++)
        {
            HashEntry* entry = this->hashtable[i];
            while(entry!=NULL)
            {
                HashEntry* prev = entry;
                entry = entry->next;
                delete prev;
            }
        }
        delete[] this->hashtable;
    }
};

#endif
