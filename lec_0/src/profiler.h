#ifndef PROFILER_H
#define PROFILER_H

/// \cond internal_docs

/**
 * \file   profiler.h
 * \author Денис Демидов ddemidov@ksu.ru
 * \brief  Класс для сбора и вывода профилирующей информации.
 */

#include <map>
#include <string>

/// Элемент профиля.
struct profile_unit {
    double tm;	    ///< Время последнего отсчета.
    double length;  ///< Общее время выполнения.

    profile_unit();
};

/// Класс для сбора и вывода профилирующей информации.
class profiler {
    public:
	/// Очистка данных профиля.
	void clear();

	/// Засекает время (устанавливает отсчет) для элемента профиля.
	/**
	 * \param key Ключ элемента профиля.
	 */
	void tic(const std::string& key);

	/// Возвращает время, прошедшее с последнего отсчета.
	/**
	 * Также увеличивает общее время выполнения элемента профиля на
	 * соответствующее значение.
	 * \param key Ключ элемента профиля.
	 * \return Время, прошедшее с последнего отсчета.
	 */
	double toc(const std::string& key);

	/// Сбрасывает общее время выполнения для элемента профиля.
	/**
	 * \param key Ключ элемента профиля.
	 */
	void reset(const std::string& key);

	/// Общее время выполнения элемента профиля.
	/**
	 * \param key Ключ элемента профиля.
	 * \return Общее время выполнения элемента профиля.
	 */
	double length(const std::string& key);

	/// Выводит на печать данные профиля.
	void report();

    private:
	/// Множество элементов профиля.
	std::map<std::string,profile_unit> unit;
};

/// \endcond
#endif
