#include <stdio.h>
#include <set>
#include <sys/timeb.h>

#include "profiler.h"

using std::map;
using std::string;
using std::set;

//---------------------------------------------------------------------------
profile_unit::profile_unit() {
    length = 0;
}

//---------------------------------------------------------------------------
void profiler::clear() {
    unit.clear();
}

//---------------------------------------------------------------------------
void profiler::tic(const string& key) {
    timeb tm;

    ftime(&tm);

    unit[key].tm = tm.time + 1e-3 * tm.millitm;
}

//---------------------------------------------------------------------------
double profiler::toc(const string& key) {
    timeb tm;

    ftime(&tm);

    double d = tm.time + 1e-3 * tm.millitm - unit[key].tm;

    unit[key].length += d;

    return d;
}

//---------------------------------------------------------------------------
void profiler::reset(const string& key) {
    unit[key].length = 0.0;
    tic(key);
}

//---------------------------------------------------------------------------
struct report_line {
    string key;
    double length;

    report_line(string k, double l) {
	key = k;
	length = l;
    }
};

//---------------------------------------------------------------------------
class cmp_lines {
    public:
	bool operator() (report_line x, report_line y) {
	    return x.length > y.length;
	}
};

//---------------------------------------------------------------------------
void profiler::report() {
    int n, max_len = 5;
    double total = 0.0;
    string format;
    map<string,profile_unit>::iterator i;
    set<report_line, cmp_lines> rep;
    set<report_line, cmp_lines>::iterator j;

    for(i = unit.begin(); i != unit.end(); i++) {
	int buf = (int)i->first.length();
	if (buf > max_len) max_len = buf;
	total += i->second.length;

	rep.insert(report_line(i->first, i->second.length));
    }

    printf("\n");
    for(j = rep.begin(); j != rep.end(); j++) {
	format = "[%s:";
	for(n = (int)j->key.length(); n <= max_len; n++) format += " ";
	format += "%8.2lf sec.] (%5.2lf%%)\n";
	printf(format.c_str(), j->key.c_str(), j->length,
		100.0 * j->length / total);
    }

    format = "[";
    for(n = 0; n <= max_len + 14; n++) format += "-";
    format += "]\n";
    printf(format.c_str());

    format = "[TOTAL:";
    for(n = 5; n <= max_len; n++) format += " ";
    format += "%8.2lf sec.]\n\n";
    printf(format.c_str(), total);
}

//---------------------------------------------------------------------------
double profiler::length(const string& key) {
    return unit[key].length;
}
