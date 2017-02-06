import numpy
import plotting

arena_size = 2 ** 16


def evaluate_interval(interval, attack_size=1, vulnerable_size=1, advancing=False):
    states = []
    state_spaces = []
    locations = []

    # noinspection PyShadowingNames
    def add_state_spaces(location):
        if location not in locations:
            locations.append(location)
        locations.sort()
        spaces = [(locations[i] - locations[i - 1]) % arena_size for i in xrange(len(locations))]
        state_spaces.append(spaces)

    for t in xrange(1, arena_size + 1):
        for extra in xrange(attack_size):
            location = (vulnerable_size + interval * t + extra) % arena_size
            if advancing:
                location = (location + attack_size * (t - 1)) % arena_size

            if len(states) == 0:
                state = numpy.zeros(arena_size)
                state[location] = 1
                states.append(state)
            else:
                state = numpy.copy(states[-1])
                state[location] += 1
                states.append(state)
            add_state_spaces(location)

            if location <= vulnerable_size:
                loss = (arena_size - t) / arena_size
                loss *= numpy.mean(numpy.sum(states) / numpy.sum([1 for state in states if state > 0]))
                loss *= numpy.mean(numpy.max(state_spaces) - numpy.min(state_spaces))
                return loss, t

            locations.append(location)


spacing_graph = plotting.Graph("loss")
survival_time_graph = plotting.Graph("survival time")

top = {}


def maybe_add_to_top(i, distribution, time):
    score = (arena_size - time) * distribution
    score_to_interval = {val[2]: key for key, val in top.iteritems()}
    if len(top) == 0 or score < numpy.max(score_to_interval.keys()):
        if len(top) >= 5:
            top.pop(score_to_interval[sorted(score_to_interval.keys())[-1]])
        top[i] = (distribution, time, score)
        print 'Top five:'
        for interval, scores in sorted(top.items(), key=lambda x: x[1][2]):
            print '\t%d (%s) - %f %d %f' % (interval, hex(interval), scores[0], scores[1], scores[2])


def main(attack_size=1, vulnerable_size=1, advancing=False):
    for i in xrange(1, arena_size):
        loss, time = evaluate_interval(i, attack_size, vulnerable_size, advancing)

        spacing_graph.maybe_add(i, i, loss)
        survival_time_graph.maybe_add(i, i, time)
        plotting.replot()

        maybe_add_to_top(i, loss, time)


if __name__ == '__main__':
    main(4, 6, True)
